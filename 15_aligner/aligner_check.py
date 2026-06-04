"""
aligner_check.py — Visual sanity-check for a single per-trial .npz file.

Produces three PNG files:
    <stem>_ppg_plot.png        — 4 PPG channels in a 2×2 grid  (first N s)
    <stem>_ultrasound_plot.png — 2 ultrasound heatmaps stacked  (first N s)
    <stem>_ear_plot.png        — EAR + eye-closed shading        (first N s)

Usage:
    python aligner_check.py path/to/p2_2.npz
    python aligner_check.py path/to/p2_2.npz --seconds 30 --output_dir ./plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed; safe for headless runs too
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Default sample rates ──────────────────────────────────────────────────────
PPG_FS = 25      # Hz
US_FS  = 100     # Hz
EAR_FS = 60      # fps


# ── Shared helpers ────────────────────────────────────────────────────────────

def _trim(arr: np.ndarray, fs: float, seconds: float):
    """Return (time_axis_s, signal) clipped to the first `seconds` of data."""
    n = min(len(arr), int(seconds * fs))
    return np.arange(n) / fs, arr[:n]


def _no_data(ax, title: str) -> None:
    ax.set_title(f"{title}\n(no data)", fontsize=10)
    ax.axis("off")


# ── PPG figure ────────────────────────────────────────────────────────────────

def plot_ppg(data: dict, stem: str, output_dir: Path,
             seconds: float, ppg_fs: float) -> Path:
    panels = [
        ("washout_finger_ppg", "Washout — Finger PPG"),
        ("washout_glass_ppg",  "Washout — Glass PPG"),
        ("video_finger_ppg",   "Video — Finger PPG"),
        ("video_glass_ppg",    "Video — Glass PPG"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"{stem}  ·  PPG  (first {seconds:.0f} s)", fontsize=13)

    for ax, (key, title) in zip(axes.flat, panels):
        arr = data.get(key)
        if arr is None or arr.size == 0:
            _no_data(ax, title)
            continue
        t, sig = _trim(arr, ppg_fs, seconds)
        ax.plot(t, sig, linewidth=0.8, color="steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("IR value")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    out = output_dir / f"{stem}_ppg_plot.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Ultrasound figure ─────────────────────────────────────────────────────────

def _us_heatmap(ax, arr2d: np.ndarray, title: str,
                seconds: float, us_fs: float) -> None:
    """Draw the first `seconds` of `arr2d` (channels × samples) as a heatmap."""
    n = min(arr2d.shape[1], int(seconds * us_fs))
    data = arr2d[:, :n]
    vmax = np.percentile(np.abs(data), 99) or 1.0
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[0, n / us_fs, -0.5, data.shape[0] - 0.5],
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, label="Amplitude", pad=0.02)


def plot_ultrasound(data: dict, stem: str, output_dir: Path,
                    seconds: float, us_fs: float) -> Path:
    panels = [
        ("washout_ultrasound", "Washout — Ultrasound heatmap"),
        ("video_ultrasound",   "Video   — Ultrasound heatmap"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"{stem}  ·  Ultrasound  (first {seconds:.0f} s)", fontsize=13)

    for ax, (key, title) in zip(axes, panels):
        raw = data.get(key)
        if raw is None or raw.size == 0:
            _no_data(ax, title)
            continue
        arr2d = np.squeeze(raw)            # (1, 100, N) → (100, N)
        if arr2d.ndim != 2:
            ax.set_title(f"{title}\n(unexpected shape {raw.shape})", fontsize=10)
            ax.axis("off")
            continue
        _us_heatmap(ax, arr2d, title, seconds, us_fs)

    plt.tight_layout()
    out = output_dir / f"{stem}_ultrasound_plot.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── EAR figure ────────────────────────────────────────────────────────────────

def plot_ear(data: dict, stem: str, output_dir: Path,
             seconds: float, ear_fs: float) -> Path:
    panels = [
        ("washout_ear", "washout_blink", "Washout — EAR"),
        ("video_ear",   "video_blink",   "Video   — EAR"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f"{stem}  ·  Eye Aspect Ratio  (first {seconds:.0f} s)", fontsize=13)

    for ax, (ear_key, blink_key, title) in zip(axes, panels):
        ear_arr   = data.get(ear_key)
        blink_arr = data.get(blink_key)

        if ear_arr is None or ear_arr.size == 0:
            _no_data(ax, title)
            continue

        t, ear = _trim(ear_arr, ear_fs, seconds)
        n = len(t)

        # shade frames where eye is closed (blink == 1)
        if blink_arr is not None and blink_arr.size >= n:
            closed = blink_arr[:n].astype(bool)
            ax.fill_between(
                t, 0, 1,
                where=closed,
                transform=ax.get_xaxis_transform(),
                color="salmon", alpha=0.35,
                label="eye closed",
            )

        ax.plot(t, ear, linewidth=0.8, color="steelblue", label="EAR")
        ax.axhline(0.21, color="tomato", linewidth=0.8,
                   linestyle="--", label="threshold 0.21")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EAR")
        ax.set_ylim(0, 0.55)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = output_dir / f"{stem}_ear_plot.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Per-file processor ────────────────────────────────────────────────────────

def process_one(npz_path: Path, output_dir: Path,
                seconds: float, ppg_fs: float,
                us_fs: float, ear_fs: float) -> None:
    stem = npz_path.stem
    print(f"\n{'─'*60}")
    print(f"Processing: {npz_path.name}")

    with np.load(npz_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}

    for k, v in data.items():
        print(f"  {k:30s}: shape={v.shape}  dtype={v.dtype}")

    plot_ppg(data, stem, output_dir, seconds, ppg_fs)
    plot_ultrasound(data, stem, output_dir, seconds, us_fs)
    plot_ear(data, stem, output_dir, seconds, ear_fs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual sanity-check for all .npz files in a folder "
                    "(or a single .npz file)."
    )
    parser.add_argument("input",
                        help="Path to a .npz file  OR  a folder of .npz files")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: same folder as each npz)")
    parser.add_argument("--seconds",  type=float, default=20.0,
                        help="How many seconds to plot (default: 20)")
    parser.add_argument("--ppg_fs",   type=float, default=PPG_FS,
                        help=f"PPG sample rate Hz (default: {PPG_FS})")
    parser.add_argument("--us_fs",    type=float, default=US_FS,
                        help=f"Ultrasound sample rate Hz (default: {US_FS})")
    parser.add_argument("--ear_fs",   type=float, default=EAR_FS,
                        help=f"EAR frame rate fps (default: {EAR_FS})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Collect npz files
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
            process_one(npz_path, out_dir,
                        args.seconds, args.ppg_fs, args.us_fs, args.ear_fs)
        except Exception as e:
            print(f"  [ERROR] {npz_path.name}: {e}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(npz_files)} file(s).")


if __name__ == "__main__":
    main()
