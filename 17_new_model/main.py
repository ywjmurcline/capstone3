"""
Shared utilities for multi-modal emotion classification.
Supports modalities: 'ppg', 'ear', 'ultrasound'

Data structure (windowed NPZ format):
    windows_dir/p{participant}_{trial_index}.npz
    Each NPZ has arrays shaped (n_windows, ...) per modality key.
    Ground truth is loaded from trials.json.
"""

import os
import csv
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

PPG_LENGTH = 250            # samples per window
EAR_LENGTH = 600            # samples per window
US_SHAPE   = (100, 1000)    # (spatial_bins, time_samples) per window

VALID_MODALITIES = ('ppg', 'ear', 'ultrasound')
VALID_GT_TYPES   = ('valence', 'arousal', 'va', 'emotion')

ALL_EMOTION_TAGS = ('amusing', 'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad')

BATCH_SIZE        = 32
NUM_EPOCHS        = 50
LEARNING_RATE     = 1e-3
SMOOTHING_STRIDES = [2, 3]
DOWNSAMPLE_RATES  = [2, 3]

# V/A binarization thresholds
# Valence: 1-3 → Low(0), 4 → excluded, 5-7 → High(1)
# Arousal: 1-4 → Low(0), 5-7 → High(1)
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


# ─── Confirmation prompt ─────────────────────────────────────────────────────

def print_confirm(dataset: 'MultiModalDataset', extra: dict = None) -> None:
    """
    Print experiment configuration in yellow and wait for the user to type Y.
    extra can contain: epochs, batch, lr, val_split, cv_folds, work_dir.
    Exits the process if the user does not confirm.
    """
    Y  = '\033[93m'   # bright yellow
    B  = '\033[1m'    # bold
    R  = '\033[0m'    # reset
    W  = 56
    ex = extra or {}

    def sep():
        print(Y + '─' * W + R)

    def row(label: str, value: str):
        print(f"  {Y}{B}{label:<22}{R}{Y}{value}{R}")

    print()
    sep()
    print(f"  {Y}{B}EXPERIMENT CONFIGURATION{R}")
    sep()

    row("Participant(s)",   ", ".join(dataset.participant_ids) or "unknown")
    row("GT type",          dataset.gt_type)
    row("Classes",          "  ".join(f"[{i}] {n}" for i, n in enumerate(dataset.class_names)))

    print(f"  {Y}{'':─<52}{R}")

    row("Modalities",       ", ".join(dataset.modalities))
    if 'ppg' in dataset.modalities:
        row("  PPG array",  dataset.ppg_key)
    if 'ear' in dataset.modalities:
        row("  EAR array",  dataset.ear_key)
    if 'ultrasound' in dataset.modalities:
        row("  US  array",  dataset.us_key)

    print(f"  {Y}{'':─<52}{R}")

    # Distribution table
    hdr = f"{'Class':<20}{'Before':>10}{'After':>10}{'Kept %':>8}"
    print(f"  {Y}{B}{hdr}{R}")
    total_before = sum(dataset.raw_counts.values())
    total_after  = sum(dataset.final_counts.values())
    for i, name in enumerate(dataset.class_names):
        before = dataset.raw_counts.get(i, 0)
        after  = dataset.final_counts.get(i, 0)
        pct    = f"{100 * after / before:.0f}%" if before else "—"
        print(f"  {Y}[{i}] {name:<17}{before:>10}{after:>10}{pct:>8}{R}")
    print(f"  {Y}{'Total':<20}{total_before:>10}{total_after:>10}{'':>8}{R}")

    print(f"  {Y}{'':─<52}{R}")

    if 'epochs' in ex:
        row("Epochs",       str(ex['epochs']))
        row("Batch size",   str(ex['batch']))
        row("Learning rate", str(ex['lr']))
    if 'cv_folds' in ex:
        row("CV folds",     str(ex['cv_folds']))
    row("Val split",        str(ex.get('val_split', '—')))
    if 'work_dir' in ex:
        row("Output dir",   ex['work_dir'])

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
            'mode':            'single',
            'participant_ids': dataset.participant_ids,
            'gt_type':         dataset.gt_type,
            'class_names':     dataset.class_names,
            'modalities':      dataset.modalities,
            'ppg_key':         dataset.ppg_key if 'ppg'        in dataset.modalities else None,
            'ear_key':         dataset.ear_key if 'ear'        in dataset.modalities else None,
            'us_key':          dataset.us_key  if 'ultrasound' in dataset.modalities else None,
            'distribution':    {
                str(i): {
                    'name':   name,
                    'before': dataset.raw_counts.get(i, 0),
                    'after':  dataset.final_counts.get(i, 0),
                }
                for i, name in enumerate(dataset.class_names)
            },
            **{k: v for k, v in ex.items() if k != 'work_dir'},
        }
        with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)


# ─── Train / val split ───────────────────────────────────────────────────────

def make_balanced_split(samples: list, val_split: float, seed: int = 42):
    """Stratified split. samples = [(label, npz_path, win_idx), ...]"""
    rng = np.random.default_rng(seed)
    class_to_idx: dict = {}
    for i, sample in enumerate(samples):
        lbl = sample[0]
        class_to_idx.setdefault(lbl, []).append(i)

    train_idx, val_idx = [], []
    for lbl in sorted(class_to_idx):
        idxs  = rng.permutation(np.array(class_to_idx[lbl]))
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return train_idx, val_idx


# ─── Ground truth resolution ─────────────────────────────────────────────────

def _valence_bin(v: float):
    """Returns 0 (Low), 1 (High), or None (V=4, excluded)."""
    v = int(v)
    if v <= _V_LOW_MAX:
        return 0
    if v >= _V_HIGH_MIN:
        return 1
    return None


def _arousal_bin(a: float):
    """Returns 0 (Low) or 1 (High)."""
    return 0 if int(a) <= _A_LOW_MAX else 1


def _dominant_emotion(emotion_tag: dict):
    """Returns the dominant emotion name, or None on tie.
    Tie-break: neutral vs exactly one non-neutral → pick non-neutral."""
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
    """
    Returns (label_int, class_name_list) or (None, None) if trial is excluded.

    gt_type='valence' : 2-class binary
    gt_type='arousal' : 2-class binary
    gt_type='va'      : 4-class (V_bin*2 + A_bin); trials with V=4 excluded
    gt_type='emotion' : n-class; only emotion_classes subset is used
    """
    gt = trial.get('ground_truth', {})

    if gt_type == 'valence':
        v = gt.get('valence')
        if v is None:
            return None, None
        lbl = _valence_bin(v)
        return lbl, ['Low_V', 'High_V']

    if gt_type == 'arousal':
        a = gt.get('arousal')
        if a is None:
            return None, None
        return _arousal_bin(a), ['Low_A', 'High_A']

    if gt_type == 'va':
        v = gt.get('valence')
        a = gt.get('arousal')
        if v is None or a is None:
            return None, None
        v_lbl = _valence_bin(v)
        if v_lbl is None:
            return None, None  # V=4 excluded
        a_lbl = _arousal_bin(a)
        return v_lbl * 2 + a_lbl, ['LowV_LowA', 'LowV_HighA', 'HighV_LowA', 'HighV_HighA']

    if gt_type == 'emotion':
        if not emotion_classes:
            raise ValueError("emotion_classes must be specified for gt_type='emotion'")
        dominant = _dominant_emotion(gt.get('emotion_tag', {}))
        if dominant is None or dominant not in emotion_classes:
            return None, None
        return emotion_classes.index(dominant), list(emotion_classes)

    raise ValueError(f"Unknown gt_type '{gt_type}'. Valid: {VALID_GT_TYPES}")


# ─── Undersampling ────────────────────────────────────────────────────────────

def _undersample_to_minority(windows_by_class: dict) -> dict:
    """
    Balance classes to the minority count, preferring trailing windows
    from each trial (proportional allocation across trials within a class).

    windows_by_class[label] = [(trial_id, win_idx, npz_path), ...]
    """
    min_count = min(len(v) for v in windows_by_class.values())
    result = {}

    for label, windows in windows_by_class.items():
        if len(windows) <= min_count:
            result[label] = list(windows)
            continue

        # Group by trial
        groups: dict = defaultdict(list)
        trial_path: dict = {}
        for (tid, win_idx, npz_path) in windows:
            groups[tid].append(win_idx)
            trial_path[tid] = npz_path
        for tid in groups:
            groups[tid].sort()

        total      = len(windows)
        ratio      = min_count / total
        trial_ids  = sorted(groups.keys())
        allocs     = {tid: max(0, round(len(groups[tid]) * ratio)) for tid in trial_ids}

        # Exact adjustment to hit min_count
        diff = sum(allocs.values()) - min_count
        if diff > 0:
            for tid in sorted(trial_ids, key=lambda t: -allocs[t]):
                if diff == 0:
                    break
                if allocs[tid] > 0:
                    allocs[tid] -= 1
                    diff -= 1
        elif diff < 0:
            diff = -diff
            for tid in sorted(trial_ids, key=lambda t: -(len(groups[t]) - allocs[t])):
                if diff == 0:
                    break
                if allocs[tid] < len(groups[tid]):
                    allocs[tid] += 1
                    diff -= 1

        kept = []
        for tid in trial_ids:
            n_take = allocs[tid]
            if n_take > 0:
                for w in groups[tid][-n_take:]:
                    kept.append((tid, w, trial_path[tid]))

        result[label] = kept

    return result


# ─── Per-participant window loader ───────────────────────────────────────────

def _load_participant_windows(
    windows_dir: str,
    trials_json: str,
    gt_type: str,
    emotion_classes: list = None,
) -> tuple:
    """
    Load raw window lists for one participant directory.

    Returns:
        windows_by_class   : {label: [(trial_id, win_idx, npz_path), ...]}
        class_name_by_label: {label: str}
        participant_ids    : [str, ...]  (from manifest)
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


# ─── Per-participant 3-way split ──────────────────────────────────────────────

def _split_participant(
    windows_by_class: dict,
    val_split: float,
    test_split: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Shuffle within each class (using rng) then 3-way split.
    Split order: train | val | test  (test comes from the end so it's always
    the latest windows, consistent with the trailing-window preference).

    Returns (train_list, val_list, test_list) where entries are
        (label, npz_path, win_idx).
    """
    train_l, val_l, test_l = [], [], []
    for label in sorted(windows_by_class.keys()):
        wins    = list(windows_by_class[label])
        indices = rng.permutation(len(wins))
        wins    = [wins[i] for i in indices]
        n       = len(wins)

        n_test  = max(1, round(n * test_split))
        n_val   = max(1, round((n - n_test) * val_split))
        n_train = n - n_test - n_val

        if n_train < 1:
            logging.warning(
                f"[Split] Class {label}: only {n} windows after undersampling "
                f"— all assigned to test."
            )
            for (tid, win_idx, npz_path) in wins:
                test_l.append((label, npz_path, win_idx))
            continue

        for (tid, win_idx, npz_path) in wins[:n_train]:
            train_l.append((label, npz_path, win_idx))
        for (tid, win_idx, npz_path) in wins[n_train:n_train + n_val]:
            val_l.append((label, npz_path, win_idx))
        for (tid, win_idx, npz_path) in wins[n_train + n_val:]:
            test_l.append((label, npz_path, win_idx))

    return train_l, val_l, test_l


# ─── Dataset ─────────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Loads windowed .npz files with ground truth from trials.json.

    Each NPZ in windows_dir contains arrays shaped (n_windows, ...) per key.
    Ground truth is resolved per trial from trials.json.

    Construct via __init__ (single-participant) or _make() (pre-built samples).

    __getitem__ returns (data_dict, label, identifier)
      data_dict maps modality → float32 tensor:
        'ppg'        → (PPG_LENGTH,)              = (250,)
        'ear'        → (EAR_LENGTH,)              = (600,)
        'ultrasound' → (US_SHAPE[0], US_SHAPE[1]) = (100, 1000)

    Public attributes:
        samples         : [(label, npz_path, win_idx), ...]
        num_classes     : int
        class_names     : [str, ...]
        raw_counts      : {label: int}   windows before undersampling
        final_counts    : {label: int}   windows after undersampling
        participant_ids : [str, ...]
    """

    _PPG_VARIANT_KEYS = {
        'finger_pre': 'video_finger_ppg_pre',
        'glass_pre':  'video_glass_ppg_pre',
        'finger':     'video_finger_ppg',
        'glass':      'video_glass_ppg',
    }
    _EAR_VARIANT_KEYS = {
        'ear':   'video_ear',    # Eye Aspect Ratio (continuous float)
        'blink': 'video_blink',  # Binary blink signal (simplified)
    }
    _US_VARIANT_KEYS = {
        'nodiff': 'video_ultrasound_nodiff',
        'diff':   'video_ultrasound_diff',
    }

    # ── single-participant constructor ────────────────────────────────────────

    def __init__(
        self,
        windows_dir: str,
        trials_json: str,
        modalities,
        gt_type: str,
        emotion_classes: list = None,
        ppg_variant: str = 'finger_pre',
        ear_variant: str = 'ear',
        us_variant: str = 'nodiff',
    ):
        for m in modalities:
            if m not in VALID_MODALITIES:
                raise ValueError(f"Unknown modality '{m}'. Valid: {VALID_MODALITIES}")
        if gt_type not in VALID_GT_TYPES:
            raise ValueError(f"Unknown gt_type '{gt_type}'. Valid: {VALID_GT_TYPES}")
        if ppg_variant not in self._PPG_VARIANT_KEYS:
            raise ValueError(f"Unknown ppg_variant '{ppg_variant}'. Valid: {list(self._PPG_VARIANT_KEYS)}")
        if ear_variant not in self._EAR_VARIANT_KEYS:
            raise ValueError(f"Unknown ear_variant '{ear_variant}'. Valid: {list(self._EAR_VARIANT_KEYS)}")
        if us_variant not in self._US_VARIANT_KEYS:
            raise ValueError(f"Unknown us_variant '{us_variant}'. Valid: {list(self._US_VARIANT_KEYS)}")

        self.modalities      = list(modalities)
        self.gt_type         = gt_type
        self.emotion_classes = list(emotion_classes) if emotion_classes else None
        self.ppg_key         = self._PPG_VARIANT_KEYS[ppg_variant]
        self.ear_key         = self._EAR_VARIANT_KEYS[ear_variant]
        self.us_key          = self._US_VARIANT_KEYS[us_variant]

        windows_by_class, class_name_by_label, pids = _load_participant_windows(
            windows_dir, trials_json, gt_type, self.emotion_classes
        )

        if not windows_by_class:
            raise RuntimeError("No valid samples found. Check gt_type, emotion_classes, and data paths.")

        balanced = _undersample_to_minority(windows_by_class)

        self.samples: list = []
        for lbl in sorted(balanced.keys()):
            for (tid, win_idx, npz_path) in balanced[lbl]:
                self.samples.append((lbl, npz_path, win_idx))

        self.num_classes     = len(balanced)
        self.class_names     = [class_name_by_label.get(i, str(i)) for i in range(self.num_classes)]
        self.raw_counts      = {lbl: len(wins) for lbl, wins in windows_by_class.items()}
        self.final_counts    = {lbl: len(wins) for lbl, wins in balanced.items()}
        self.participant_ids = pids

    # ── pre-built constructor (used by build_multi_splits) ────────────────────

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
        raw_counts: dict = None,
        final_counts: dict = None,
        participant_ids: list = None,
    ) -> 'MultiModalDataset':
        obj = super().__new__(cls)
        super(MultiModalDataset, obj).__init__()   # torch Dataset has no-op __init__
        obj.samples         = list(samples)
        obj.modalities      = list(modalities)
        obj.ppg_key         = ppg_key
        obj.ear_key         = ear_key
        obj.us_key          = us_key
        obj.num_classes     = num_classes
        obj.class_names     = list(class_names)
        obj.gt_type         = gt_type
        obj.emotion_classes = None
        obj.raw_counts      = raw_counts   or {}
        obj.final_counts    = final_counts or {}
        obj.participant_ids = list(participant_ids or [])
        return obj

    # ── data loading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, npz_path, win_idx = self.samples[idx]
        data = np.load(npz_path, allow_pickle=False)
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
                arr = arr[0]  # (1, H, W) → (H, W)
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


# ─── Multi-participant split builder ─────────────────────────────────────────

def build_multi_splits(
    participants_config: list,
    train_ids: list,
    test_config: list,
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
    Build train / val / multiple test datasets across participants.

    participants_config : [{"participant_id": ..., "directory": ...}, ...]
        directory must contain windows/ sub-dir and trials.json.
    train_ids   : [pid, ...]  participants whose data feeds train + val
    test_config : [pid | [pid, ...], ...]
        Each element defines one test set.
        A bare pid uses that participant's test split alone.
        A list of pids combines their test splits.
        PIDs here need not appear in train_ids.

    Per-participant undersampling is applied independently (each participant's
    own minority class count).  Within each participant, each class is shuffled
    before the 3-way split so the split is random but reproducible.

    Returns:
        train_ds     : MultiModalDataset  (shuffled)
        val_ds       : MultiModalDataset  (shuffled)
        test_ds_dict : {label: MultiModalDataset}
        info         : dict  (for print_confirm_multi)
    """
    # Normalise all IDs to strings
    train_ids = [str(t) for t in train_ids]
    test_config_norm = [
        [str(e) for e in entry] if isinstance(entry, list) else str(entry)
        for entry in test_config
    ]

    needed_ids = set(train_ids)
    for entry in test_config_norm:
        if isinstance(entry, list):
            needed_ids.update(entry)
        else:
            needed_ids.add(entry)

    pid_to_cfg = {str(p['participant_id']): p for p in participants_config}
    for pid in needed_ids:
        if pid not in pid_to_cfg:
            raise ValueError(f"Participant '{pid}' not found in participants_config.")

    ppg_key = MultiModalDataset._PPG_VARIANT_KEYS[ppg_variant]
    ear_key = MultiModalDataset._EAR_VARIANT_KEYS[ear_variant]
    us_key  = MultiModalDataset._US_VARIANT_KEYS[us_variant]

    # ── per-participant: load → undersample → 3-way split ────────────────────
    per_pid: dict = {}
    all_class_name_by_label: dict = {}

    for pid in sorted(needed_ids):
        base    = pid_to_cfg[pid]['directory']
        win_dir = os.path.join(base, 'windows')
        tri_js  = os.path.join(base, 'trials.json')

        logging.info(f"[MultiSplit] Participant '{pid}'  dir='{base}'")

        raw_wbc, cname_map, _ = _load_participant_windows(
            win_dir, tri_js, gt_type, emotion_classes
        )
        all_class_name_by_label.update(cname_map)

        if not raw_wbc:
            logging.warning(f"[MultiSplit] Participant '{pid}' has no valid windows — skipped.")
            per_pid[pid] = {'train': [], 'val': [], 'test': [], 'raw': {}, 'final': {}}
            continue

        balanced     = _undersample_to_minority(raw_wbc)
        raw_counts   = {lbl: len(w) for lbl, w in raw_wbc.items()}
        final_counts = {lbl: len(w) for lbl, w in balanced.items()}

        # Deterministic per-participant seed keeps results reproducible
        # across different participant orderings.
        pid_seed = seed ^ (abs(hash(pid)) % (2 ** 31))
        rng      = np.random.default_rng(pid_seed)

        train_l, val_l, test_l = _split_participant(balanced, val_split, test_split, rng)

        per_pid[pid] = {
            'train': train_l, 'val': val_l, 'test': test_l,
            'raw':   raw_counts, 'final': final_counts,
        }
        logging.info(
            f"[MultiSplit] '{pid}' → "
            f"train {len(train_l)} / val {len(val_l)} / test {len(test_l)}"
        )

    # ── canonical class ordering ─────────────────────────────────────────────
    all_labels  = sorted(all_class_name_by_label.keys())
    num_classes = len(all_labels)
    class_names = [all_class_name_by_label.get(i, str(i)) for i in range(num_classes)]

    rng_global = np.random.default_rng(seed)

    def _combine_shuffle(pid_list: list, split_key: str) -> list:
        combined = []
        for pid in pid_list:
            combined.extend(per_pid.get(pid, {}).get(split_key, []))
        idx = rng_global.permutation(len(combined)).tolist()
        return [combined[i] for i in idx]

    # ── aggregate raw / final counts for print_confirm_multi ─────────────────
    agg_raw: dict   = defaultdict(int)
    agg_final: dict = defaultdict(int)
    for pid in needed_ids:
        for lbl, cnt in per_pid[pid]['raw'].items():
            agg_raw[lbl]   += cnt
        for lbl, cnt in per_pid[pid]['final'].items():
            agg_final[lbl] += cnt

    # ── build train / val datasets ────────────────────────────────────────────
    train_samples = _combine_shuffle(train_ids, 'train')
    val_samples   = _combine_shuffle(train_ids, 'val')

    train_ds = MultiModalDataset._make(
        train_samples, modalities, ppg_key, ear_key, us_key,
        num_classes, class_names, gt_type=gt_type,
        raw_counts=dict(agg_raw), final_counts=dict(agg_final),
        participant_ids=train_ids,
    )
    val_ds = MultiModalDataset._make(
        val_samples, modalities, ppg_key, ear_key, us_key,
        num_classes, class_names, gt_type=gt_type,
        participant_ids=train_ids,
    )

    # ── build test datasets ───────────────────────────────────────────────────
    test_ds_dict: dict = {}
    test_counts:  dict = {}

    for entry in test_config_norm:
        pid_list  = entry if isinstance(entry, list) else [entry]
        ds_label  = '+'.join(pid_list)
        samples   = _combine_shuffle(pid_list, 'test')
        test_ds_dict[ds_label] = MultiModalDataset._make(
            samples, modalities, ppg_key, ear_key, us_key,
            num_classes, class_names, gt_type=gt_type,
            participant_ids=pid_list,
        )
        test_counts[ds_label] = len(samples)

    logging.info(
        f"[MultiSplit] Done — train {len(train_samples)} / val {len(val_samples)} / "
        f"tests {test_counts}"
    )

    info = {
        'gt_type':     gt_type,
        'train_ids':   train_ids,
        'test_config': test_config_norm,
        'per_pid':     per_pid,
        'class_names': class_names,
        'modalities':  modalities,
        'ppg_key':     ppg_key,
        'ear_key':     ear_key,
        'us_key':      us_key,
        'split_counts': {
            'train': len(train_samples),
            'val':   len(val_samples),
            **{f'test_{k}': v for k, v in test_counts.items()},
        },
    }
    return train_ds, val_ds, test_ds_dict, info


# ─── Multi-participant confirmation prompt ────────────────────────────────────

def print_confirm_multi(info: dict, extra: dict = None) -> None:
    """
    Print multi-participant experiment config in yellow, wait for Y.
    info is the dict returned by build_multi_splits.
    extra can contain: epochs, batch, lr, val_split, test_split, cv_folds, work_dir.
    """
    Y  = '\033[93m'
    B  = '\033[1m'
    R  = '\033[0m'
    W  = 64
    ex = extra or {}

    def sep():
        print(Y + '─' * W + R)

    def row(label: str, value: str):
        print(f"  {Y}{B}{label:<24}{R}{Y}{value}{R}")

    print()
    sep()
    print(f"  {Y}{B}MULTI-PARTICIPANT EXPERIMENT{R}")
    sep()

    row("GT type",    info['gt_type'])
    row("Classes",    "  ".join(f"[{i}] {n}" for i, n in enumerate(info['class_names'])))
    row("Train PIDs", ", ".join(info['train_ids']))

    test_labels = [
        '+'.join(e) if isinstance(e, list) else str(e)
        for e in info['test_config']
    ]
    row("Test sets",  "  |  ".join(test_labels))

    print(f"  {Y}{'':─<60}{R}")

    row("Modalities", ", ".join(info['modalities']))
    if 'ppg' in info['modalities']:
        row("  PPG array", info['ppg_key'])
    if 'ear' in info['modalities']:
        row("  EAR array", info['ear_key'])
    if 'ultrasound' in info['modalities']:
        row("  US  array", info['us_key'])

    print(f"  {Y}{'':─<60}{R}")

    # Per-participant distribution table
    class_names = info['class_names']
    nc = len(class_names)

    col_w = 12
    hdr_pid   = f"{'PID':<8}"
    hdr_cols  = "".join(f"[{i}]{class_names[i][:5]:<{col_w-3}}" for i in range(nc))
    hdr_total = f"{'Total':>{col_w}}"
    print(f"  {Y}{B}{hdr_pid}{hdr_cols}{hdr_total}  (raw→sampled){R}")

    for pid in sorted(info['per_pid'].keys()):
        d     = info['per_pid'][pid]
        raw   = d.get('raw', {})
        final = d.get('final', {})
        cols  = "".join(
            f"{str(raw.get(i,0))+'→'+str(final.get(i,0)):>{col_w}}"
            for i in range(nc)
        )
        tot   = f"{sum(raw.values())}→{sum(final.values())}"
        print(f"  {Y}{str(pid):<8}{cols}{tot:>{col_w}}{R}")

    print(f"  {Y}{'':─<60}{R}")

    sc = info.get('split_counts', {})
    row("Train windows",  str(sc.get('train', '—')))
    row("Val windows",    str(sc.get('val',   '—')))
    for k, v in sc.items():
        if k.startswith('test_'):
            row(f"Test [{k[5:]}]",   str(v))

    print(f"  {Y}{'':─<60}{R}")

    if 'epochs' in ex:
        row("Epochs",        str(ex['epochs']))
        row("Batch size",    str(ex['batch']))
        row("Learning rate", str(ex['lr']))
    if 'cv_folds' in ex:
        row("CV folds",      str(ex['cv_folds']))
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
        per_pid_counts = {
            pid: {'raw': d['raw'], 'final': d['final']}
            for pid, d in info['per_pid'].items()
        }
        config = {
            'mode':         'multi',
            'gt_type':      info['gt_type'],
            'class_names':  info['class_names'],
            'train_ids':    info['train_ids'],
            'test_config':  info['test_config'],
            'modalities':   info['modalities'],
            'ppg_key':      info['ppg_key']  if 'ppg'        in info['modalities'] else None,
            'ear_key':      info['ear_key']  if 'ear'        in info['modalities'] else None,
            'us_key':       info['us_key']   if 'ultrasound' in info['modalities'] else None,
            'per_participant': per_pid_counts,
            'split_counts': info['split_counts'],
            **{k: v for k, v in ex.items() if k != 'work_dir'},
        }
        with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
