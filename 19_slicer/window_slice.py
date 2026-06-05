"""
Cut all NPZ files in a folder into 10-second sliding windows.

Each NPZ has washout and video phases. Within each phase, all modalities are
truncated to the shortest one before windowing.

Sampling rates:
  PPG (finger, glass, pre): 25 Hz  → 250 samples / 10s window
  ear, blink:               60 Hz  → 600 samples / 10s window
  ultrasound:              100 Hz  → 1000 samples / 10s window, shape (1,100,T)

Usage:
  python window_slice.py --input_dir <dir> --output_dir <dir> [--stride 5] [--window 10]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Config: key → (npz_field_name, sample_rate, is_ultrasound)
# --------------------------------------------------------------------------- #
PHASES = ['washout', 'video']

MODALITY_CFG = {
    'finger_ppg':       (25,  False),
    'glass_ppg':        (25,  False),
    'ear':              (60,  False),
    'blink':            (60,  False),
    'finger_ppg_pre':   (25,  False),
    'glass_ppg_pre':    (25,  False),
    'ultrasound_nodiff':(100, True),
    'ultrasound_diff':  (100, True),
}

# only these modalities define the "duration" of a phase (pre is derived)
DURATION_KEYS = ['finger_ppg', 'glass_ppg', 'ear', 'blink',
                 'ultrasound_nodiff', 'ultrasound_diff']


def npz_key(phase: str, mod: str) -> str:
    return f'{phase}_{mod}'


def get_length(arr: np.ndarray, is_us: bool) -> int:
    """Return number of time samples (last axis for ultrasound, axis 0 otherwise)."""
    return arr.shape[-1] if is_us else arr.shape[0]


def truncate(arr: np.ndarray, n: int, is_us: bool) -> np.ndarray:
    return arr[..., :n] if is_us else arr[:n]


def make_windows(arr: np.ndarray, win: int, stride: int, is_us: bool) -> np.ndarray:
    """
    Slice array into windows of length `win` with step `stride`.
    Returns stacked array:  (n_windows, *spatial_dims, win)  for ultrasound
                            (n_windows, win)                  for 1-D signals
    """
    length = get_length(arr, is_us)
    starts = range(0, length - win + 1, stride)
    if is_us:
        slices = [arr[..., s:s + win] for s in starts]
    else:
        slices = [arr[s:s + win] for s in starts]
    return np.stack(slices, axis=0) if slices else np.empty((0,))


def parse_filename(stem: str) -> dict:
    """Extract participant_id and trial_index from 'p{pid}_{trial}' stem."""
    try:
        part, trial = stem.split('_', 1)
        return {'participant_id': part[1:], 'trial_index': trial}
    except Exception:
        return {'participant_id': None, 'trial_index': None}


def process_file(src_path: Path, dst_path: Path,
                 window_sec: float, stride_sec: float) -> dict:
    data = np.load(src_path, allow_pickle=True)

    out = {}
    phase_info = {}

    for phase in PHASES:
        # ------------------------------------------------------------------ #
        # 1. compute duration for each modality present in this phase
        # ------------------------------------------------------------------ #
        durations_sec = {}
        for mod in DURATION_KEYS:
            key = npz_key(phase, mod)
            if key not in data.files:
                continue
            sr, is_us = MODALITY_CFG[mod]
            n = get_length(data[key], is_us)
            durations_sec[mod] = n / sr

        if not durations_sec:
            phase_info[phase] = {'skipped': True}
            continue

        min_sec = min(durations_sec.values())

        def n_wins(dur_sec: float, sr: int) -> int:
            n = math.floor(dur_sec * sr)
            w = math.floor(window_sec * sr)
            s = math.floor(stride_sec  * sr)
            return max(0, (n - w) // s + 1)

        raw_windows = {
            mod: n_wins(dur, MODALITY_CFG[mod][0])
            for mod, dur in durations_sec.items()
        }
        phase_info[phase] = {
            'raw_durations_sec':  {k: round(v, 3) for k, v in durations_sec.items()},
            'raw_windows':        raw_windows,
            'truncated_to_sec':   round(min_sec, 3),
        }

        # ------------------------------------------------------------------ #
        # 2. window all modalities (including *_pre) using the same min_sec
        # ------------------------------------------------------------------ #
        for mod, (sr, is_us) in MODALITY_CFG.items():
            key = npz_key(phase, mod)
            if key not in data.files:
                continue

            win_samples    = math.floor(window_sec * sr)
            stride_samples = math.floor(stride_sec  * sr)
            trunc_samples  = math.floor(min_sec     * sr)

            arr      = truncate(data[key], trunc_samples, is_us)
            windows  = make_windows(arr, win_samples, stride_samples, is_us)
            out[key] = windows

        # record n_windows (same for all modalities in a phase)
        sample_key  = npz_key(phase, next(iter(durations_sec)))
        sample_mod  = next(iter(durations_sec))
        sr, is_us   = MODALITY_CFG[sample_mod]
        win_samples = math.floor(window_sec * sr)
        stride_s    = math.floor(stride_sec  * sr)
        trunc_s     = math.floor(min_sec     * sr)
        n_win       = max(0, (trunc_s - win_samples) // stride_s + 1)
        phase_info[phase]['n_windows'] = n_win

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **out)

    meta = parse_filename(src_path.stem)
    return {
        'participant_id': meta['participant_id'],
        'trial_index':    meta['trial_index'],
        'src':            str(src_path.resolve()),
        'dst':            str(dst_path.resolve()),
        'window_sec':     window_sec,
        'stride_sec':     stride_sec,
        'phases':         phase_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--window',  type=float, default=10.0, help='window length in seconds')
    parser.add_argument('--stride',  type=float, default=5.0,  help='stride in seconds')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    npz_files = sorted(input_dir.glob('*.npz'))
    if not npz_files:
        print(f'No .npz files found in {input_dir}')
        return

    for src in npz_files:
        dst = output_dir / src.name
        print(f'Processing {src.name} ...')
        info = process_file(src, dst, args.window, args.stride)
        manifest.append(info)
        for phase, pi in info['phases'].items():
            if pi.get('skipped'):
                print(f'  {phase}: skipped (no data)')
            else:
                print(f'  {phase}: truncated to {pi["truncated_to_sec"]}s'
                      f'  →  {pi["n_windows"]} windows')

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nManifest saved to {manifest_path}')


if __name__ == '__main__':
    main()
