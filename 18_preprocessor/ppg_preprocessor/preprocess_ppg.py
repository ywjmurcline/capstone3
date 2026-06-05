"""
Preprocess all *_ppg arrays in every .npz file under a given directory.
For each key ending in '_ppg', applies Butterworth bandpass filter then
robust min-max normalization, and saves the result back to the same file
under the key name + '_pre'.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ── signal processing helpers ────────────────────────────────────────────────

def butterworth_bandpass_filter(signal, fs, lowcut=0.5, highcut=5.0, order=2):
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        raise ValueError(f"highcut={highcut} >= nyquist={nyquist}; lower highcut or raise fs")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * order + 1), len(signal) - 1)
    return sosfiltfilt(sos, signal, padlen=padlen)


def robust_minmax_normalize(signal, lower_percentile=2, upper_percentile=98,
                             target_range=(-0.5, 0.5)):
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    low = np.percentile(signal, lower_percentile)
    high = np.percentile(signal, upper_percentile)
    if low == high:
        return np.zeros_like(signal)
    clipped = np.clip(signal, low, high)
    norm_01 = (clipped - low) / (high - low)
    min_t, max_t = target_range
    return norm_01 * (max_t - min_t) + min_t


def preprocess_ppg(signal, fs, lowcut=0.5, highcut=5.0, order=2,
                   lower_percentile=2, upper_percentile=98):
    filtered = butterworth_bandpass_filter(signal, fs=fs, lowcut=lowcut,
                                           highcut=highcut, order=order)
    return robust_minmax_normalize(filtered, lower_percentile, upper_percentile)


# ── per-file processing ───────────────────────────────────────────────────────

def process_npz(npz_path: Path, fs: float, dry_run: bool = False) -> list[str]:
    """Return list of keys that were processed."""
    data = np.load(npz_path, allow_pickle=False)
    ppg_keys = [k for k in data.files if k.endswith("_ppg")]

    if not ppg_keys:
        return []

    new_arrays = dict(data)  # copy all existing arrays

    processed = []
    for key in ppg_keys:
        out_key = key + "_pre"
        if out_key in new_arrays:
            print(f"  [skip] {out_key} already exists")
            continue
        try:
            result = preprocess_ppg(data[key], fs=fs)
            new_arrays[out_key] = result.astype(np.float32)
            processed.append(key)
        except Exception as e:
            print(f"  [error] {key}: {e}")

    if processed and not dry_run:
        np.savez(npz_path, **new_arrays)

    return processed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess *_ppg arrays in .npz files.")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Root directory to search for .npz files (default: current dir)")
    parser.add_argument("--fs", type=float, default=25.0,
                        help="PPG sampling frequency in Hz (default: 25)")
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=5.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--lower-pct", type=float, default=2.0,
                        help="Lower percentile for normalization (default: 2)")
    parser.add_argument("--upper-pct", type=float, default=98.0,
                        help="Upper percentile for normalization (default: 98)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process but do not write changes back to disk")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    npz_files = sorted(root.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found under {root}")
        sys.exit(0)

    print(f"Found {len(npz_files)} .npz file(s) under {root}")
    print(f"Settings: fs={args.fs} Hz, bandpass=[{args.lowcut}, {args.highcut}] Hz, "
          f"order={args.order}, norm percentiles=[{args.lower_pct}, {args.upper_pct}]")
    if args.dry_run:
        print("[DRY RUN — no files will be modified]")
    print()

    total_processed = 0
    for npz_path in npz_files:
        print(f"{npz_path.relative_to(root)}")
        processed = process_npz(npz_path, fs=args.fs, dry_run=args.dry_run)
        if processed:
            for k in processed:
                print(f"  + {k}_pre")
            total_processed += len(processed)
        else:
            print("  (no new _ppg arrays to process)")

    print(f"\nDone. {total_processed} array(s) preprocessed across {len(npz_files)} file(s).")


if __name__ == "__main__":
    main()
