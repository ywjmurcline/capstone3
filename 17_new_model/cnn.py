"""
Multi-modal CNN for emotion classification.
==========================================
Modality-specific feature extractors (each outputs 128-dim):
  PPG (250 samples)    : dual-scale 1D CNN  — Branch1 (kernel 3) + Branch2 (kernel 9, dilated)
  EAR (500 samples)    : single 1D CNN      — 3 × Conv(3, stride 2)
  Ultrasound (100×1000): 2D CNN             — 3 × Conv2d(3, stride 2)
Active modalities are controlled by --modalities; final features concat → FC → softmax.
"""

import os
import sys
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from main import (
    MultiModalDataset, make_balanced_split,
    build_multi_splits, print_confirm, print_confirm_multi,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    setup_logging, save_epoch_json,
    plot_confusion_matrix, compute_f1,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _dpad(kernel: int, dilation: int) -> int:
    """Padding to keep spatial size approximately the same under dilation."""
    return dilation * (kernel - 1) // 2


# ─── Feature extractors ──────────────────────────────────────────────────────

class PPGFeatureExtractor(nn.Module):
    """
    Dual-scale 1D CNN for PPG. Input: (B, 250).

    Branch1 (kernel=3, dilation=1):
        Conv(3,32,s2) → Conv(3,64,s2) → Conv(3,128,s1) → 63×128
    Branch2 (kernel=9, dilation {2,4,8}):
        Conv(9,32,s2,d2) → Conv(9,64,s2,d4) → Conv(9,128,s1,d8) → 63×128
    Fusion:
        Concat → 63×256 → Conv(1,128) → GAP → 128-dim
    """

    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=9, stride=2,
                      padding=_dpad(9, 2), dilation=2, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2,
                      padding=_dpad(9, 4), dilation=4, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=1,
                      padding=_dpad(9, 8), dilation=8, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = x.unsqueeze(1).float()                          # (B,1,250)
        b1 = self.branch1(x)                                 # (B,128,L1)
        b2 = self.branch2(x)                                 # (B,128,L2)
        L  = min(b1.size(-1), b2.size(-1))
        feat = torch.cat([b1[..., :L], b2[..., :L]], dim=1) # (B,256,L)
        feat = self.fusion(feat)                              # (B,128,L)
        return self.gap(feat).squeeze(-1)                     # (B,128)


class EARFeatureExtractor(nn.Module):
    """
    1D CNN for EAR signal. Input: (B, 500).
    Conv(3,32,s2) → Conv(3,64,s2) → Conv(3,128,s2) → GAP → 128-dim.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).float()          # (B,1,500)
        return self.gap(self.net(x)).squeeze(-1)  # (B,128)


class UltrasoundFeatureExtractor(nn.Module):
    """
    2D CNN for ultrasound. Input: (B, 100, 1000) → treated as (B,1,100,1000).
    Conv2d(3,32,s2) → Conv2d(3,64,s2) → Conv2d(3,128,s2) → GAP → 128-dim.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = x.unsqueeze(1).float()         # (B,1,100,1000)
        feat = self.net(x)                     # (B,128,H',W')
        return self.gap(feat).flatten(1)       # (B,128)


# ─── Full model ──────────────────────────────────────────────────────────────

class MultiModalCNN(nn.Module):
    """
    Multi-modal CNN. Active extractors are determined by `modalities`.
    Each extractor outputs 128-dim. All active outputs are concatenated
    then fed into a 3-layer FC classifier.
    """

    _FEAT_DIM = 128

    def __init__(self, modalities: list, num_classes: int):
        super().__init__()
        self.modalities = list(modalities)

        if 'ppg'        in modalities: self.ppg_ext = PPGFeatureExtractor()
        if 'ear'        in modalities: self.ear_ext = EARFeatureExtractor()
        if 'ultrasound' in modalities: self.us_ext  = UltrasoundFeatureExtractor()

        feat_dim = self._FEAT_DIM * len(modalities)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, data_dict: dict) -> torch.Tensor:
        feats = []
        if 'ppg'        in self.modalities: feats.append(self.ppg_ext(data_dict['ppg']))
        if 'ear'        in self.modalities: feats.append(self.ear_ext(data_dict['ear']))
        if 'ultrasound' in self.modalities: feats.append(self.us_ext(data_dict['ultrasound']))
        return self.classifier(torch.cat(feats, dim=1))


# ─── Training / evaluation ───────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data_dict, labels, _ in loader:
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        labels    = labels.to(device)
        optimizer.zero_grad()
        logits    = model(data_dict)
        loss      = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.detach().argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_details=False):
    """
    Returns (avg_loss, accuracy) or (avg_loss, accuracy, details).
    details = [(basename, true_label, pred_label), ...]
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    details = []
    for data_dict, labels, basenames in loader:
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        labels    = labels.to(device)
        logits    = model(data_dict)
        loss      = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds      = logits.argmax(dim=1)
        correct   += (preds == labels).sum().item()
        total     += len(labels)
        if return_details:
            for bn, p, l in zip(basenames, preds.cpu().tolist(), labels.cpu().tolist()):
                details.append((bn, l, p))
    avg_loss = total_loss / total
    acc      = correct / total
    if return_details:
        return avg_loss, acc, details
    return avg_loss, acc


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-modal CNN. '
                    'Use --participants_json for cross-participant mode, '
                    'or windows_dir + --trials_json for single-participant mode.'
    )
    # ── single-participant positional (optional in multi mode) ────────────────
    parser.add_argument('windows_dir', nargs='?', default='',
                        help='[Single mode] Windowed .npz dir with manifest.json.')
    parser.add_argument('--trials_json',  default='',
                        help='[Single mode] Path to trials.json.')
    # ── multi-participant ─────────────────────────────────────────────────────
    parser.add_argument('--participants_json', default='',
                        help='[Multi mode] Path to participants JSON: '
                             '[{"participant_id":..., "directory":...}, ...]')
    parser.add_argument('--train_participants', default='',
                        help='[Multi mode] JSON list of participant IDs for train+val, '
                             'e.g. \'["mym","ywj"]\'.')
    parser.add_argument('--test_participants', default='',
                        help='[Multi mode] JSON list of test configs, e.g. '
                             '\'["1","2",["1","2"]]\'. Each entry is one test set.')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='[Multi mode] Fraction held out for test per participant.')
    # ── shared ────────────────────────────────────────────────────────────────
    parser.add_argument('--gt_type',      required=True,
                        choices=['valence', 'arousal', 'va', 'emotion'])
    parser.add_argument('--emotion_classes', default='',
                        help='Comma-separated emotion classes (gt_type=emotion only).')
    parser.add_argument('--ppg_variant',  default='finger_pre',
                        choices=['finger_pre', 'glass_pre', 'finger', 'glass'],
                        help='PPG: finger_pre/glass_pre (preprocessed) or finger/glass (raw).')
    parser.add_argument('--ear_variant',  default='ear',
                        choices=['ear', 'blink'],
                        help='EAR: ear (Eye Aspect Ratio) or blink (binary).')
    parser.add_argument('--us_variant',   default='nodiff',
                        choices=['nodiff', 'diff'],
                        help='Ultrasound: nodiff or diff (frame-differenced).')
    parser.add_argument('--modalities',   default='ppg',
                        help='Comma-separated: ppg,ear,ultrasound')
    parser.add_argument('--work_dir',     required=True)
    parser.add_argument('--epochs',    type=int,   default=NUM_EPOCHS)
    parser.add_argument('--batch',     type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',        type=float, default=LEARNING_RATE)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]
    emotion_classes = (
        [e.strip() for e in args.emotion_classes.split(',') if e.strip()]
        if args.emotion_classes else None
    )
    os.makedirs(args.work_dir, exist_ok=True)
    log_path = os.path.join(args.work_dir, 'cnn_train.log')
    setup_logging(log_path)
    logging.info(f"[Run] Multi-modal CNN  work_dir='{args.work_dir}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[Config] Device: {device}  Modalities: {modalities}  GT: {args.gt_type}")

    multi_mode    = bool(args.participants_json)
    test_ds_dict  = {}   # only populated in multi mode

    # ── dataset construction ──────────────────────────────────────────────────
    if multi_mode:
        with open(args.participants_json) as f:
            participants_config = json.load(f)
        if not args.train_participants:
            parser.error('--train_participants is required in multi mode.')
        if not args.test_participants:
            parser.error('--test_participants is required in multi mode.')
        train_ids   = json.loads(args.train_participants)
        test_config = json.loads(args.test_participants)

        train_ds, val_ds, test_ds_dict, info = build_multi_splits(
            participants_config = participants_config,
            train_ids           = train_ids,
            test_config         = test_config,
            modalities          = modalities,
            gt_type             = args.gt_type,
            emotion_classes     = emotion_classes,
            ppg_variant         = args.ppg_variant,
            ear_variant         = args.ear_variant,
            us_variant          = args.us_variant,
            val_split           = args.val_split,
            test_split          = args.test_split,
        )
        print_confirm_multi(info, {
            'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
            'val_split': args.val_split, 'test_split': args.test_split,
            'work_dir': args.work_dir,
        })
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names

    else:
        if not args.windows_dir or not args.trials_json:
            parser.error('windows_dir and --trials_json are required in single mode.')
        dataset = MultiModalDataset(
            windows_dir     = args.windows_dir,
            trials_json     = args.trials_json,
            modalities      = modalities,
            gt_type         = args.gt_type,
            emotion_classes = emotion_classes,
            ppg_variant     = args.ppg_variant,
            ear_variant     = args.ear_variant,
            us_variant      = args.us_variant,
        )
        print_confirm(dataset, {
            'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
            'val_split': args.val_split, 'work_dir': args.work_dir,
        })
        train_idx, val_idx = make_balanced_split(dataset.samples, args.val_split)
        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        num_classes = dataset.num_classes
        class_names = dataset.class_names
        logging.info(f"[Dataset] Train: {len(train_idx)}  Val: {len(val_idx)}")

    logging.info(f"[Config] Num classes: {num_classes}  {class_names}")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    model = MultiModalCNN(modalities, num_classes).to(device)
    logging.info(
        f"[Model] Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── training loop ─────────────────────────────────────────────────────────
    model_path    = os.path.join(args.work_dir, 'cnn_best.pt')
    best_val_acc  = 0.0
    best_f1_macro = 0.0
    epoch_details = {}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, details = evaluate(
            model, val_loader, criterion, device, return_details=True)
        scheduler.step()

        true_labels = [d[1] for d in details]
        pred_labels = [d[2] for d in details]
        f1_info     = compute_f1(true_labels, pred_labels, num_classes)

        epoch_details[str(epoch)] = {
            'val_loss':  round(val_loss, 4),
            'val_acc':   round(val_acc,  4),
            **f1_info,
            'correct':   [bn for bn, t, p in details if t == p],
            'incorrect': [bn for bn, t, p in details if t != p],
        }
        logging.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"f1={f1_info['f1_macro']:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_f1_macro = f1_info['f1_macro']
            torch.save(model.state_dict(), model_path)
            logging.info(f"  ✓ New best val_acc={best_val_acc:.4f}  f1_macro={best_f1_macro:.4f}")

    logging.info(f"\nTraining done. Best val_acc={best_val_acc:.4f}  f1_macro={best_f1_macro:.4f}")
    save_epoch_json(epoch_details, os.path.join(args.work_dir, 'val_epoch_details.json'))

    # ── final evaluation on best model ────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))

    def _eval_and_save(loader, ds_label, tag):
        _, _, det = evaluate(model, loader, criterion, device, return_details=True)
        tl = [d[1] for d in det]
        pl = [d[2] for d in det]
        fi = compute_f1(tl, pl, num_classes)
        labels_range = list(range(num_classes))
        cm = sk_confusion_matrix(tl, pl, labels=labels_range)
        cm_path = os.path.join(args.work_dir, f'confusion_matrix_{tag}.png')
        plot_confusion_matrix(cm, class_names, cm_path, title=f'Confusion Matrix — {ds_label}')
        return fi

    # val set (used for model selection)
    val_f1 = _eval_and_save(val_loader, 'Val', 'val')

    # test sets (multi mode: one per test config entry; single mode: reuse val)
    test_results = {}
    if test_ds_dict:
        for ds_label, test_ds in test_ds_dict.items():
            test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)
            fi = _eval_and_save(test_loader, f'Test {ds_label}', f'test_{ds_label}')
            test_results[ds_label] = fi
            logging.info(
                f"[Test {ds_label}] f1_macro={fi['f1_macro']:.4f} "
                f"f1_weighted={fi['f1_weighted']:.4f}"
            )
    else:
        test_results['val'] = val_f1

    summary = {
        'modalities':   modalities,
        'num_classes':  num_classes,
        'best_val_acc': round(best_val_acc, 4),
        'val_f1':       val_f1,
        'test_results': test_results,
    }
    summary_path = os.path.join(args.work_dir, 'results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"[Output] Results summary → '{summary_path}'")
    logging.info("[Run] Done.")
