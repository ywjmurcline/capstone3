#!/usr/bin/env bash
# Usage: ./run_preprocess_ppg.sh [npz_directory] [--dry-run]
# Defaults to the current directory if no path is given.

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# NPZ_DIR="${1:-.}"
# SHIFT_COUNT=1

# # Pass remaining flags (e.g. --dry-run) through to the Python script
# shift $SHIFT_COUNT 2>/dev/null || true

python3 "preprocess_ppg.py" \
    "/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/qishui/npz" \
    --fs 25 \
    --lowcut 0.5 \
    --highcut 5.0 \
    --order 2 \
    --lower-pct 2 \
    --upper-pct 98 \
    "$@"
