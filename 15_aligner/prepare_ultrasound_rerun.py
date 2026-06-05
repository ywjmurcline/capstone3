#!/usr/bin/env python3
"""
prepare_ultrasound_rerun.py — Migrate progress.json and strip old ultrasound
arrays from NPZ files so aligner_data_1.py can re-process them cleanly.

What this script does:
  1. Reads progress.json and replaces the old single boolean keys
       ultrasound_washout / ultrasound_video
     with four new keys (all set to False):
       ultrasound_washout_nodiff, ultrasound_washout_diff
       ultrasound_video_nodiff,   ultrasound_video_diff
  2. For every NPZ listed in progress, removes the old arrays
       washout_ultrasound, video_ultrasound
     and leaves all other arrays (PPG, ear, _pre, …) untouched.

Use --dry_run first to preview changes without modifying any file.

Usage:
    python prepare_ultrasound_rerun.py \\
        --progress /path/to/npz/progress.json \\
        --npz_dir  /path/to/npz/ \\
        [--dry_run]
"""

import argparse
import json
from pathlib import Path

import numpy as np

OLD_US_PROGRESS_KEYS = {"ultrasound_washout", "ultrasound_video"}
NEW_US_PROGRESS_KEYS = [
    "ultrasound_washout_nodiff",
    "ultrasound_washout_diff",
    "ultrasound_video_nodiff",
    "ultrasound_video_diff",
]
OLD_NPZ_KEYS = {"washout_ultrasound", "video_ultrasound"}


def migrate_entry(entry: dict) -> dict:
    for k in OLD_US_PROGRESS_KEYS:
        entry.pop(k, None)
    for k in NEW_US_PROGRESS_KEYS:
        entry.setdefault(k, False)
    return entry


def strip_ultrasound_from_npz(npz_path: Path, dry_run: bool) -> None:
    with np.load(npz_path) as f:
        kept    = {k: f[k] for k in f.files if k not in OLD_NPZ_KEYS}
        removed = [k for k in f.files if k in OLD_NPZ_KEYS]

    if not removed:
        print(f"  [skip ] {npz_path.name} — no old ultrasound arrays found")
        return

    if dry_run:
        print(f"  [dry  ] {npz_path.name} — would remove: {removed}  keep: {list(kept)}")
        return

    np.savez(str(npz_path), **kept)
    print(f"  [strip] {npz_path.name} — removed: {removed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate progress.json + strip old ultrasound from NPZ files"
    )
    parser.add_argument("--progress", required=True,
                        help="Path to progress.json")
    parser.add_argument("--npz_dir",  required=True,
                        help="Directory containing the .npz files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview changes without writing anything")
    args = parser.parse_args()

    progress_path = Path(args.progress)
    npz_dir       = Path(args.npz_dir)

    if not progress_path.exists():
        raise FileNotFoundError(f"progress.json not found: {progress_path}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"npz_dir is not a directory: {npz_dir}")

    with open(progress_path) as f:
        progress: dict = json.load(f)

    # ── Migrate progress.json ────────────────────────────────────────────────
    for entry in progress.values():
        migrate_entry(entry)

    if args.dry_run:
        print("=== DRY RUN: progress.json would become ===")
        print(json.dumps(progress, indent=2, ensure_ascii=False))
    else:
        progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False))
        print(f"Updated: {progress_path}")

    # ── Strip NPZ files ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Stripping ultrasound arrays from NPZ files …\n")
    for entry in progress.values():
        npz_name = entry.get("npz")
        if not npz_name:
            continue
        npz_path = npz_dir / npz_name
        if not npz_path.exists():
            print(f"  [miss ] {npz_name} — file not found, skipping")
            continue
        strip_ultrasound_from_npz(npz_path, dry_run=args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete — nothing written.")


if __name__ == "__main__":
    main()
