"""
check_ppg_pre.py — Plot the 4 preprocessed PPG arrays in each .npz file.

Produces one PNG per file:
    <stem>_ppg_pre_plot.png  — 2×2 grid of *_ppg_pre channels

Usage:
    python check_ppg_pre.py path/to/p2_2.npz
    python check_ppg_pre.py path/to/npz_folder/
    python check_ppg_pre.py path/to/npz_folder/ --seconds 30 --output_dir ./plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PPG_FS = 25  # Hz


def _trim(arr: np.ndarray, fs: float, seconds: float):
    n = min(len(arr), int(seconds * fs))
    return np.arange(n) / fs, arr[:n]


def _no_data(ax, title: str) -> None:
    ax.set_title(f"{title}\n(no data)", fontsize=10)
    ax.axis("off")


def plot_ppg_pre(data: dict, stem: str, output_dir: Path,
                 seconds: float, ppg_fs: float) -> Path:
    panels = [
        ("washout_finger_ppg_pre", "Washout — Finger PPG (pre)"),
        ("washout_glass_ppg_pre",  "Washout — Glass PPG (pre)"),
        ("video_finger_ppg_pre",   "Video — Finger PPG (pre)"),
        ("video_glass_ppg_pre",    "Video — Glass PPG (pre)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"{stem}  ·  Preprocessed PPG  (first {seconds:.0f} s)", fontsize=13)

    for ax, (key, title) in zip(axes.flat, panels):
        arr = data.get(key)
        if arr is None or arr.size == 0:
            _no_data(ax, title)
            continue
        t, sig = _trim(arr, ppg_fs, seconds)
        ax.plot(t, sig, linewidth=0.8, color="steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (normalized)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    out = output_dir / f"{stem}_ppg_pre_plot.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def process_one(npz_path: Path, output_dir: Path,
                seconds: float, ppg_fs: float) -> None:
    stem = npz_path.stem
    print(f"\n{'─'*60}")
    print(f"Processing: {npz_path.name}")

    with np.load(npz_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}

    pre_keys = [k for k in data if k.endswith("_ppg_pre")]
    if not pre_keys:
        print("  [skip] no *_ppg_pre arrays found — run preprocess_ppg.py first")
        return

    plot_ppg_pre(data, stem, output_dir, seconds, ppg_fs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot preprocessed PPG channels from .npz files."
    )
    parser.add_argument("input",
                        help="Path to a .npz file OR a folder of .npz files")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: same folder as each npz)")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="How many seconds to plot (default: 20)")
    parser.add_argument("--ppg_fs", type=float, default=PPG_FS,
                        help=f"PPG sample rate Hz (default: {PPG_FS})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        npz_files = sorted(input_path.glob("*.npz"))
        if not npz_files:
            print(f"No .npz files found in: {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        npz_files = [input_path]

    print(f"Found {len(npz_files)} .npz file(s) to process.")

    for npz_path in npz_files:
        out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            process_one(npz_path, out_dir, args.seconds, args.ppg_fs)
        except Exception as e:
            print(f"  [ERROR] {npz_path.name}: {e}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(npz_files)} file(s).")


if __name__ == "__main__":
    main()
