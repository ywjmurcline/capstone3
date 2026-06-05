"""
Shared utilities for multi-modal emotion classification.
Supports modalities: 'ppg', 'ear', 'ultrasound'

Data structure (windowed NPZ format):
    windows_dir/p{participant}_{trial_index}.npz
    Each NPZ has arrays shaped (n_windows, ...) per modality key.
    Ground truth is loaded from trials.json.

Split strategy (new in 22_models):
    For each participant, for each class:
        shuffle all windows → 60-20-20 (train-val-test)
    Then combine across participants and shuffle each split globally.
    No undersampling.
"""

import os
import json
import logging
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

# ─── Constants ───────────────────────────────────────────────────────────────

PPG_LENGTH = 250
EAR_LENGTH = 600
US_SHAPE   = (100, 1000)

VALID_MODALITIES = ('ppg', 'ear', 'ultrasound')
VALID_GT_TYPES   = ('valence', 'arousal', 'va', 'emotion')

ALL_EMOTION_TAGS = ('amusing', 'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad')

BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3

_V_LOW_MAX  = 3
_V_NEUTRAL  = 4
_V_HIGH_MIN = 5
_A_LOW_MAX  = 4
_A_HIGH_MIN = 5


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Output helpers ──────────────────────────────────────────────────────────

def save_epoch_json(epoch_details: dict, json_path: str) -> None:
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(epoch_details, f, indent=2)
    logging.info(f"[Output] Epoch details JSON → '{json_path}'")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    title: str = 'Confusion Matrix',
) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"[Output] Confusion matrix → '{save_path}'")


# ─── F1 helpers ──────────────────────────────────────────────────────────────

def compute_f1(true_labels: list, pred_labels: list, num_classes: int) -> dict:
    labels   = list(range(num_classes))
    macro    = f1_score(true_labels, pred_labels, average='macro',
                        labels=labels, zero_division=0)
    weighted = f1_score(true_labels, pred_labels, average='weighted',
                        labels=labels, zero_division=0)
    per_cls  = f1_score(true_labels, pred_labels, average=None,
                        labels=labels, zero_division=0)
    return {
        'f1_macro':     round(float(macro), 4),
        'f1_weighted':  round(float(weighted), 4),
        'f1_per_class': [round(float(v), 4) for v in per_cls],
    }


# ─── Ground truth resolution ─────────────────────────────────────────────────

def _valence_bin(v: float):
    v = int(v)
    if v <= _V_LOW_MAX:  return 0
    if v >= _V_HIGH_MIN: return 1
    return None


def _arousal_bin(a: float):
    return 0 if int(a) <= _A_LOW_MAX else 1


def _dominant_emotion(emotion_tag: dict):
    if not emotion_tag:
        return None
    max_val = max(emotion_tag.values())
    if max_val == 0:
        return None
    candidates = [k for k, v in emotion_tag.items() if v == max_val]
    if len(candidates) == 1:
        return candidates[0]
    non_neutral = [k for k in candidates if k != 'neutral']
    if len(non_neutral) == 1:
        return non_neutral[0]
    return None


def _resolve_label(trial: dict, gt_type: str, emotion_classes: list = None):
    gt = trial.get('ground_truth', {})

    if gt_type == 'valence':
        v = gt.get('valence')
        if v is None: return None, None
        return _valence_bin(v), ['Low_V', 'High_V']

    if gt_type == 'arousal':
        a = gt.get('arousal')
        if a is None: return None, None
        return _arousal_bin(a), ['Low_A', 'High_A']

    if gt_type == 'va':
        v, a = gt.get('valence'), gt.get('arousal')
        if v is None or a is None: return None, None
        v_lbl = _valence_bin(v)
        if v_lbl is None: return None, None
        return v_lbl * 2 + _arousal_bin(a), ['LowV_LowA', 'LowV_HighA', 'HighV_LowA', 'HighV_HighA']

    if gt_type == 'emotion':
        if not emotion_classes:
            raise ValueError("emotion_classes must be specified for gt_type='emotion'")
        dominant = _dominant_emotion(gt.get('emotion_tag', {}))
        if dominant is None or dominant not in emotion_classes:
            return None, None
        return emotion_classes.index(dominant), list(emotion_classes)

    raise ValueError(f"Unknown gt_type '{gt_type}'. Valid: {VALID_GT_TYPES}")


# ─── Per-participant window loader ───────────────────────────────────────────

def _load_participant_windows(
    windows_dir: str,
    trials_json: str,
    gt_type: str,
    emotion_classes: list = None,
) -> tuple:
    """
    Returns:
        windows_by_class   : {label: [(trial_id, win_idx, npz_path), ...]}
        class_name_by_label: {label: str}
        participant_ids    : [str, ...]
    """
    manifest_path = os.path.join(windows_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(trials_json) as f:
        trials_by_index = {str(t['index']): t for t in json.load(f)}

    windows_by_class: dict    = defaultdict(list)
    class_name_by_label: dict = {}
    participant_ids: set      = set()

    for entry in manifest:
        trial_index = str(entry['trial_index'])
        dst_path    = entry['dst']
        n_windows   = entry['phases']['video']['n_windows']
        if 'participant_id' in entry:
            participant_ids.add(str(entry['participant_id']))

        trial = trials_by_index.get(trial_index)
        if trial is None or trial.get('error'):
            continue

        label, lbl_names = _resolve_label(trial, gt_type, emotion_classes)
        if label is None:
            continue

        class_name_by_label[label] = lbl_names[label] if label < len(lbl_names) else str(label)
        for win_idx in range(n_windows):
            windows_by_class[label].append((trial_index, win_idx, dst_path))

    return dict(windows_by_class), class_name_by_label, sorted(participant_ids)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Loads windowed .npz files with ground truth from trials.json.

    __getitem__ returns (data_dict, label, identifier)
      data_dict maps modality → float32 tensor:
        'ppg'        → (PPG_LENGTH,)
        'ear'        → (EAR_LENGTH,)
        'ultrasound' → (US_SHAPE[0], US_SHAPE[1])
    """

    _PPG_VARIANT_KEYS = {
        'finger_pre': 'video_finger_ppg_pre',
        'glass_pre':  'video_glass_ppg_pre',
        'finger':     'video_finger_ppg',
        'glass':      'video_glass_ppg',
    }
    _EAR_VARIANT_KEYS = {
        'ear':   'video_ear',
        'blink': 'video_blink',
    }
    _US_VARIANT_KEYS = {
        'nodiff': 'video_ultrasound_nodiff',
        'diff':   'video_ultrasound_diff',
    }

    @classmethod
    def _make(
        cls,
        samples: list,
        modalities: list,
        ppg_key: str,
        ear_key: str,
        us_key: str,
        num_classes: int,
        class_names: list,
        gt_type: str = '',
        participant_ids: list = None,
    ) -> 'MultiModalDataset':
        obj = super().__new__(cls)
        super(MultiModalDataset, obj).__init__()
        obj.samples         = list(samples)
        obj.modalities      = list(modalities)
        obj.ppg_key         = ppg_key
        obj.ear_key         = ear_key
        obj.us_key          = us_key
        obj.num_classes     = num_classes
        obj.class_names     = list(class_names)
        obj.gt_type         = gt_type
        obj.emotion_classes = None
        obj.participant_ids = list(participant_ids or [])
        return obj

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, npz_path, win_idx = self.samples[idx]
        data   = np.load(npz_path, allow_pickle=False)
        result = {}

        if 'ppg' in self.modalities:
            sig = data[self.ppg_key][win_idx].astype(np.float32).flatten()[:PPG_LENGTH]
            if len(sig) < PPG_LENGTH:
                sig = np.pad(sig, (0, PPG_LENGTH - len(sig)))
            s = sig.std()
            if s > 1e-8:
                sig = (sig - sig.mean()) / s
            result['ppg'] = torch.from_numpy(sig)

        if 'ear' in self.modalities:
            sig = data[self.ear_key][win_idx].astype(np.float32).flatten()[:EAR_LENGTH]
            if len(sig) < EAR_LENGTH:
                sig = np.pad(sig, (0, EAR_LENGTH - len(sig)))
            s = sig.std()
            if s > 1e-8:
                sig = (sig - sig.mean()) / s
            result['ear'] = torch.from_numpy(sig)

        if 'ultrasound' in self.modalities:
            arr = data[self.us_key][win_idx].astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            H, W = US_SHAPE
            if arr.shape[0] < H:
                arr = np.pad(arr, ((0, H - arr.shape[0]), (0, 0)))
            arr = arr[:H]
            if arr.shape[1] < W:
                arr = np.pad(arr, ((0, 0), (0, W - arr.shape[1])))
            arr = arr[:, :W]
            s = arr.std()
            if s > 1e-8:
                arr = (arr - arr.mean()) / s
            result['ultrasound'] = torch.from_numpy(arr)

        trial_id   = os.path.splitext(os.path.basename(npz_path))[0]
        identifier = f"{trial_id}_w{win_idx:03d}"
        return result, label, identifier


# ─── Split builder ────────────────────────────────────────────────────────────

def build_splits(
    participants_cfg: list,
    modalities: list,
    gt_type: str,
    emotion_classes: list = None,
    ppg_variant: str = 'finger_pre',
    ear_variant: str = 'ear',
    us_variant: str = 'nodiff',
    val_split: float = 0.2,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    For each participant, for each class:
        shuffle all windows with a per-participant seed
        split sequentially: first 60% → train, next 20% → val, last 20% → test
    Combine across participants, then shuffle each split with the global seed.

    participants_cfg entries must have:
        'participant_id': str
        'windows_dir':   str   (direct path to windowed NPZ directory)
        'trials_json':   str   (path to trials.json)

    Returns (train_ds, val_ds, test_ds, info).
    """
    ppg_key = MultiModalDataset._PPG_VARIANT_KEYS[ppg_variant]
    ear_key = MultiModalDataset._EAR_VARIANT_KEYS[ear_variant]
    us_key  = MultiModalDataset._US_VARIANT_KEYS[us_variant]

    all_train:  list = []
    all_val:    list = []
    all_test:   list = []
    all_class_name_by_label: dict = {}
    participant_ids: list = []
    per_pid_info:    dict = {}

    for p_cfg in participants_cfg:
        pid      = str(p_cfg['participant_id'])
        win_dir  = p_cfg['windows_dir']
        tri_js   = p_cfg['trials_json']
        participant_ids.append(pid)

        raw_wbc, cname_map, _ = _load_participant_windows(
            win_dir, tri_js, gt_type, emotion_classes
        )
        all_class_name_by_label.update(cname_map)

        if not raw_wbc:
            logging.warning(f"[Split] Participant '{pid}' has no valid windows — skipped.")
            per_pid_info[pid] = {}
            continue

        pid_seed = seed ^ (abs(hash(pid)) % (2 ** 31))
        pid_rng  = np.random.default_rng(pid_seed)

        pid_counts: dict = {'total': {}, 'train': {}, 'val': {}, 'test': {}}

        for label in sorted(raw_wbc.keys()):
            wins = list(raw_wbc[label])          # [(trial_id, win_idx, npz_path), ...]
            perm = pid_rng.permutation(len(wins))
            wins = [wins[i] for i in perm]       # shuffle per person per class

            n       = len(wins)
            n_val   = max(1, round(n * val_split))
            n_test  = max(1, round(n * test_split))
            n_train = n - n_val - n_test

            pid_counts['total'][label] = n

            if n_train < 1:
                logging.warning(
                    f"[Split] '{pid}' class {label}: only {n} windows — all → test."
                )
                for (tid, win_idx, npz_path) in wins:
                    all_test.append((label, npz_path, win_idx))
                pid_counts['train'][label] = 0
                pid_counts['val'][label]   = 0
                pid_counts['test'][label]  = n
                continue

            for (tid, win_idx, npz_path) in wins[:n_train]:
                all_train.append((label, npz_path, win_idx))
            for (tid, win_idx, npz_path) in wins[n_train:n_train + n_val]:
                all_val.append((label, npz_path, win_idx))
            for (tid, win_idx, npz_path) in wins[n_train + n_val:]:
                all_test.append((label, npz_path, win_idx))

            pid_counts['train'][label] = n_train
            pid_counts['val'][label]   = n_val
            pid_counts['test'][label]  = n - n_train - n_val

        per_pid_info[pid] = pid_counts

        cls_summary = "  ".join(
            f"cls{lbl}({pid_counts['total'][lbl]})="
            f"{pid_counts['train'].get(lbl,0)}/"
            f"{pid_counts['val'].get(lbl,0)}/"
            f"{pid_counts['test'].get(lbl,0)}"
            for lbl in sorted(raw_wbc.keys())
        )
        logging.info(f"[Split] '{pid}' — {cls_summary}")

    # Shuffle each split globally
    global_rng = np.random.default_rng(seed)
    all_train = [all_train[i] for i in global_rng.permutation(len(all_train))]
    all_val   = [all_val[i]   for i in global_rng.permutation(len(all_val))]
    all_test  = [all_test[i]  for i in global_rng.permutation(len(all_test))]

    # Class info
    all_labels  = sorted(all_class_name_by_label.keys())
    num_classes = (max(all_labels) + 1) if all_labels else 0
    class_names = [all_class_name_by_label.get(i, str(i)) for i in range(num_classes)]

    def _count_by_class(samples: list) -> dict:
        c: dict = defaultdict(int)
        for (lbl, _, _) in samples:
            c[lbl] += 1
        return dict(c)

    info = {
        'gt_type':        gt_type,
        'class_names':    class_names,
        'modalities':     modalities,
        'ppg_key':        ppg_key,
        'ear_key':        ear_key,
        'us_key':         us_key,
        'participant_ids': participant_ids,
        'per_pid':        per_pid_info,
        'split_counts': {
            'train': len(all_train),
            'val':   len(all_val),
            'test':  len(all_test),
        },
        'class_counts': {
            'train': _count_by_class(all_train),
            'val':   _count_by_class(all_val),
            'test':  _count_by_class(all_test),
        },
    }

    train_ds = MultiModalDataset._make(
        all_train, modalities, ppg_key, ear_key, us_key,
        num_classes, class_names, gt_type=gt_type,
        participant_ids=participant_ids,
    )
    val_ds = MultiModalDataset._make(
        all_val, modalities, ppg_key, ear_key, us_key,
        num_classes, class_names, gt_type=gt_type,
        participant_ids=participant_ids,
    )
    test_ds = MultiModalDataset._make(
        all_test, modalities, ppg_key, ear_key, us_key,
        num_classes, class_names, gt_type=gt_type,
        participant_ids=participant_ids,
    )

    return train_ds, val_ds, test_ds, info


# ─── Confirmation prompt ─────────────────────────────────────────────────────

def print_confirm(info: dict, extra: dict = None) -> None:
    """
    Print experiment config with per-class counts, wait for Y.
    info is the dict returned by build_splits.
    extra: epochs, batch, lr, val_split, test_split, work_dir.
    """
    Y  = '\033[93m'
    B  = '\033[1m'
    R  = '\033[0m'
    W  = 64
    ex = extra or {}

    def sep():
        print(Y + '─' * W + R)

    def row(label: str, value: str):
        print(f"  {Y}{B}{label:<26}{R}{Y}{value}{R}")

    print()
    sep()
    print(f"  {Y}{B}EXPERIMENT CONFIGURATION{R}")
    sep()

    row("GT type",        info['gt_type'])
    row("Classes",        "  ".join(f"[{i}] {n}" for i, n in enumerate(info['class_names'])))
    row("Participants",   ", ".join(info['participant_ids']))
    row("Modalities",     ", ".join(info['modalities']))
    if 'ppg' in info['modalities']:
        row("  PPG array",  info['ppg_key'])
    if 'ear' in info['modalities']:
        row("  EAR array",  info['ear_key'])
    if 'ultrasound' in info['modalities']:
        row("  US  array",  info['us_key'])

    print(f"  {Y}{'':─<60}{R}")

    # Per-class count table
    class_names  = info['class_names']
    class_counts = info['class_counts']
    hdr = f"  {'Split':<8}" + "".join(f"[{i}]{class_names[i][:8]:<11}" for i in range(len(class_names))) + f"{'Total':>8}"
    print(f"  {Y}{B}{hdr}{R}")
    for split in ('train', 'val', 'test'):
        cnt   = class_counts.get(split, {})
        cols  = "".join(f"{cnt.get(i, 0):>12}" for i in range(len(class_names)))
        total = sum(cnt.values())
        print(f"  {Y}{split:<8}{cols}{total:>8}{R}")

    print(f"  {Y}{'':─<60}{R}")

    # Per-participant breakdown
    per_pid = info.get('per_pid', {})
    if per_pid:
        print(f"  {Y}{B}  Per-participant (total / train / val / test per class):{R}")
        for pid in sorted(per_pid.keys()):
            d = per_pid[pid]
            lines = []
            for lbl in sorted(d.get('total', {}).keys()):
                lines.append(
                    f"cls{lbl}:{d['total'].get(lbl,0)}"
                    f"→{d['train'].get(lbl,0)}/"
                    f"{d['val'].get(lbl,0)}/"
                    f"{d['test'].get(lbl,0)}"
                )
            print(f"  {Y}  {pid:<10}{' | '.join(lines)}{R}")

    print(f"  {Y}{'':─<60}{R}")

    if 'epochs' in ex:
        row("Epochs",        str(ex['epochs']))
        row("Batch size",    str(ex['batch']))
        row("Learning rate", str(ex['lr']))
    vs = ex.get('val_split', '—')
    ts = ex.get('test_split', '—')
    row("Val / Test split", f"{vs} / {ts}")
    if 'work_dir' in ex:
        row("Output dir",    ex['work_dir'])

    sep()

    try:
        ans = input(f"  {Y}{B}Proceed? [Y/n]: {R}")
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Aborted.")

    if ans.strip().lower() not in ('y', 'yes'):
        raise SystemExit("Aborted.")
    print()

    work_dir = ex.get('work_dir')
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        config = {
            'gt_type':         info['gt_type'],
            'class_names':     info['class_names'],
            'modalities':      info['modalities'],
            'ppg_key':         info['ppg_key'] if 'ppg'        in info['modalities'] else None,
            'ear_key':         info['ear_key'] if 'ear'        in info['modalities'] else None,
            'us_key':          info['us_key']  if 'ultrasound' in info['modalities'] else None,
            'participant_ids': info['participant_ids'],
            'split_counts':    info['split_counts'],
            'class_counts':    info['class_counts'],
            **{k: v for k, v in ex.items() if k != 'work_dir'},
        }
        with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
