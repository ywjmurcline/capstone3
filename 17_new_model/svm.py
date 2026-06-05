"""
Multi-modal SVM for emotion classification.
============================================
Feature extraction per modality:
  PPG  (250 samples)    : time + frequency features from 5 views
                          (original / 2 smoothed / 2 downsampled) → 85-dim
  EAR  (500 samples)    : same multi-view approach, fs=30 Hz      → 85-dim
  Ultrasound (100×1000) : mean-signal features + per-row stats    → 227-dim
Active modalities controlled by --modalities; features concatenated → SVM (RBF + grid search).
"""

import os
import sys
import json
import logging
import argparse
import warnings
import pickle

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from main import (
    MultiModalDataset, build_multi_splits, print_confirm, print_confirm_multi,
    PPG_LENGTH, EAR_LENGTH, US_SHAPE,
    SMOOTHING_STRIDES, DOWNSAMPLE_RATES,
    setup_logging, save_epoch_json,
    plot_confusion_matrix, compute_f1,
)


# ─── Shared time / frequency feature extraction ──────────────────────────────

def time_features(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    zcr = np.sum(np.diff(np.sign(x)) != 0) / max(len(x) - 1, 1)
    mad = np.mean(np.abs(x - x.mean()))
    return np.array([
        x.mean(), x.std(), x.min(), x.max(),
        x.max() - x.min(),
        rms, float(stats.skew(x)), float(stats.kurtosis(x)),
        zcr, mad,
    ])


def freq_features(x: np.ndarray, fs: float = 25.0) -> np.ndarray:
    N    = len(x)
    mag  = np.abs(rfft(x)) / N
    freq = rfftfreq(N, d=1.0 / fs)

    if mag.sum() < 1e-10:
        return np.zeros(7)

    dom_freq = freq[np.argmax(mag)]
    centroid = np.sum(freq * mag) / (mag.sum() + 1e-10)

    bands = [(0, 1), (1, 3), (3, 8), (8, 12)]
    band_power = []
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < hi)
        band_power.append(float(mag[mask].sum()) if mask.any() else 0.0)

    p    = mag / (mag.sum() + 1e-10)
    sent = -np.sum(p * np.log(p + 1e-12))

    return np.array([dom_freq, centroid] + band_power + [float(sent)])


def _multiview_1d(signal: np.ndarray, fs: float) -> np.ndarray:
    """5-view (original + 2 smoothed + 2 downsampled) time+freq features."""
    views = [signal]
    for s in SMOOTHING_STRIDES:
        kernel = np.ones(s) / s
        views.append(np.convolve(signal, kernel, mode='same'))
    for d in DOWNSAMPLE_RATES:
        ds = signal[::d]
        up = np.interp(
            np.arange(len(signal)),
            np.linspace(0, len(signal) - 1, len(ds)),
            ds,
        )
        views.append(up)
    parts = []
    for v in views:
        parts.append(time_features(v))
        parts.append(freq_features(v, fs=fs))
    return np.concatenate(parts)   # 5 × 17 = 85 features


# ─── Per-modality feature extraction ─────────────────────────────────────────

def extract_ppg_features(signal: np.ndarray) -> np.ndarray:
    return _multiview_1d(signal, fs=25.0)


def extract_ear_features(signal: np.ndarray) -> np.ndarray:
    return _multiview_1d(signal, fs=30.0)


def extract_ultrasound_features(arr: np.ndarray) -> np.ndarray:
    """
    Features for 2D ultrasound array (H, W) = (100, 1000):
      1. Mean signal (mean across H rows): time (10) + freq (7) = 17
      2. Per-row mean and std:             H + H                 = 200
      3. 10 temporal bin means:                                    10
    Total: 227 features.
    """
    if arr.ndim == 3:
        arr = arr[0]
    H, W = arr.shape

    mean_sig = arr.mean(axis=0)                      # (W,)
    feat_sig = np.concatenate([
        time_features(mean_sig),
        freq_features(mean_sig, fs=1.0),
    ])

    row_means = arr.mean(axis=1)                     # (H,)
    row_stds  = arr.std(axis=1)                      # (H,)

    n_bins    = 10
    bin_means = []
    for i in range(n_bins):
        lo = i * W // n_bins
        hi = (i + 1) * W // n_bins
        bin_means.append(float(arr[:, lo:hi].mean()))

    return np.concatenate([feat_sig, row_means, row_stds, np.array(bin_means)])


# ─── Feature matrix builder ──────────────────────────────────────────────────

def build_feature_matrix(dataset: MultiModalDataset, modalities: list):
    """
    Load each sample window, extract features for requested modalities, concat.
    Returns (X: float32 array, y: int array, identifiers: list[str]).
    """
    X, y, names = [], [], []
    for (label, npz_path, win_idx) in dataset.samples:
        data = np.load(npz_path, allow_pickle=False)
        feat_parts = []

        if 'ppg' in modalities:
            sig = data[dataset.ppg_key][win_idx].astype(np.float32).flatten()[:PPG_LENGTH]
            if len(sig) < PPG_LENGTH:
                sig = np.pad(sig, (0, PPG_LENGTH - len(sig)))
            s = sig.std()
            if s > 1e-8:
                sig = (sig - sig.mean()) / s
            feat_parts.append(extract_ppg_features(sig))

        if 'ear' in modalities:
            sig = data[dataset.ear_key][win_idx].astype(np.float32).flatten()[:EAR_LENGTH]
            if len(sig) < EAR_LENGTH:
                sig = np.pad(sig, (0, EAR_LENGTH - len(sig)))
            s = sig.std()
            if s > 1e-8:
                sig = (sig - sig.mean()) / s
            feat_parts.append(extract_ear_features(sig))

        if 'ultrasound' in modalities:
            arr = data[dataset.us_key][win_idx].astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]  # (1, H, W) → (H, W)
            H_t, W_t = US_SHAPE
            if arr.shape[0] < H_t:
                arr = np.pad(arr, ((0, H_t - arr.shape[0]), (0, 0)))
            arr = arr[:H_t]
            if arr.shape[1] < W_t:
                arr = np.pad(arr, ((0, 0), (0, W_t - arr.shape[1])))
            arr = arr[:, :W_t]
            s = arr.std()
            if s > 1e-8:
                arr = (arr - arr.mean()) / s
            feat_parts.append(extract_ultrasound_features(arr))

        if not feat_parts:
            raise RuntimeError(
                f"No features extracted from '{npz_path}' window {win_idx}. "
                f"Check that requested modalities {modalities} are present."
            )

        trial_id   = os.path.splitext(os.path.basename(npz_path))[0]
        X.append(np.concatenate(feat_parts))
        y.append(label)
        names.append(f"{trial_id}_w{win_idx:03d}")

    return np.array(X, dtype=np.float32), np.array(y), names


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-modal SVM. '
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
                             '\'["1","2",["1","2"]]\'.')
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
    parser.add_argument('--val_split',  type=float, default=0.2)
    parser.add_argument('--cv_folds',   type=int,   default=5)
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]
    emotion_classes = (
        [e.strip() for e in args.emotion_classes.split(',') if e.strip()]
        if args.emotion_classes else None
    )
    os.makedirs(args.work_dir, exist_ok=True)
    log_path = os.path.join(args.work_dir, 'svm_train.log')
    setup_logging(log_path)
    logging.info(f"[Run] Multi-modal SVM  work_dir='{args.work_dir}'")

    multi_mode   = bool(args.participants_json)
    test_ds_dict = {}

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
            'val_split': args.val_split, 'test_split': args.test_split,
            'cv_folds': args.cv_folds, 'work_dir': args.work_dir,
        })
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names

        # In multi mode: train on train_ds, select hyperparams on val_ds,
        # then evaluate on each test set.
        logging.info(f"[Feature] Extracting train features ({len(train_ds.samples)} samples)...")
        X_train, y_train, _ = build_feature_matrix(train_ds, modalities)
        logging.info(f"[Feature] Extracting val features ({len(val_ds.samples)} samples)...")
        X_val,   y_val,   _ = build_feature_matrix(val_ds, modalities)

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
            'val_split': args.val_split, 'cv_folds': args.cv_folds,
            'work_dir': args.work_dir,
        })
        num_classes = dataset.num_classes
        class_names = dataset.class_names

        logging.info(f"[Feature] Extracting from {len(dataset.samples)} samples ...")
        X_all, y_all, names_all = build_feature_matrix(dataset, modalities)
        logging.info(f"[Feature] Feature vector length: {X_all.shape[1]}")

        indices = np.arange(len(y_all))
        train_idx, test_idx = train_test_split(
            indices, test_size=args.val_split, stratify=y_all, random_state=42
        )
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val,   y_val   = X_all[test_idx],  y_all[test_idx]
        test_ds_dict['val'] = None   # sentinel: use X_val / y_val directly
        logging.info(f"[Data] Train: {len(y_train)}  Val/Test: {len(y_val)}")

    logging.info(f"[Feature] Feature vector length: {X_train.shape[1]}")
    logging.info(f"[Config] Classes: {class_names}")

    # ── grid search (fit on train, scored on val for multi; CV for single) ────
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', class_weight='balanced')),
    ])
    param_grid = {
        'svm__C':     [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.001],
    }
    cv = StratifiedKFold(
        n_splits=min(args.cv_folds, len(y_train)), shuffle=True, random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search = GridSearchCV(pipe, param_grid, cv=cv,
                              scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)

    logging.info(f"[SVM] Best params : {search.best_params_}")
    logging.info(f"[SVM] CV accuracy : {search.best_score_:.4f}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    def _svm_eval_and_save(X_t, y_t, names_t, ds_label, tag):
        y_pred = search.predict(X_t)
        acc    = accuracy_score(y_t, y_pred)
        fi     = compute_f1(y_t.tolist(), y_pred.tolist(), num_classes)
        report = classification_report(
            y_t, y_pred, labels=list(range(num_classes)), target_names=class_names
        )
        logging.info(f"[SVM {ds_label}] acc={acc:.4f}  f1_macro={fi['f1_macro']:.4f}")
        logging.info("\n" + report)
        labels_range = list(range(num_classes))
        cm = sk_confusion_matrix(y_t, y_pred, labels=labels_range)
        cm_path = os.path.join(args.work_dir, f'confusion_matrix_{tag}.png')
        plot_confusion_matrix(cm, class_names, cm_path, title=f'Confusion Matrix — {ds_label}')
        return acc, fi, y_pred, names_t

    # val
    val_acc, val_f1, val_pred, val_names = _svm_eval_and_save(
        X_val, y_val, [], 'Val', 'val'
    )

    # test sets
    test_results = {}
    if multi_mode:
        for ds_label, test_ds in test_ds_dict.items():
            logging.info(f"[Feature] Extracting test features '{ds_label}' ({len(test_ds.samples)} samples)...")
            X_t, y_t, n_t = build_feature_matrix(test_ds, modalities)
            acc, fi, _, _ = _svm_eval_and_save(X_t, y_t, n_t, f'Test {ds_label}', f'test_{ds_label}')
            test_results[ds_label] = {'acc': round(float(acc), 4), **fi}
    else:
        test_results['val'] = {'acc': round(float(val_acc), 4), **val_f1}

    # ── save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.work_dir, 'svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':       search.best_estimator_,
            'num_classes': num_classes,
            'class_names': class_names,
            'modalities':  modalities,
            'gt_type':     args.gt_type,
        }, f)
    logging.info(f"[Output] Model → '{model_path}'")

    summary = {
        'modalities':   modalities,
        'num_classes':  num_classes,
        'cv_best_acc':  round(float(search.best_score_), 4),
        'val_f1':       val_f1,
        'test_results': test_results,
    }
    summary_path = os.path.join(args.work_dir, 'results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"[Output] Results summary → '{summary_path}'")
    logging.info("[Run] Done.")
