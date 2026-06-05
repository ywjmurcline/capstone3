#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/lily/Documents/myApps/Capstone_Saveme/venv-psychopy/bin/python"
PY_SCRIPT="$SCRIPT_DIR/analyze_ground_truth.py"

TRIALS="${1:-$SCRIPT_DIR/data/mym/trials.json}"
MANIFEST="${2:-$SCRIPT_DIR/data/mym/windows/manifest.json}"
OUTPUT="${3:-$(dirname "$TRIALS")/ground_truth_stats}"

echo "trials  : $TRIALS"
echo "manifest: $MANIFEST"
echo "output  : $OUTPUT"

"$PYTHON" "$PY_SCRIPT" "$TRIALS" "$MANIFEST" "$OUTPUT"
