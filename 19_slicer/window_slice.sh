#!/bin/bash
set -e

INPUT_DIR="/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/mym/npz"
OUTPUT_DIR="/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/mym/windows"
WINDOW=10
STRIDE=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/window_slice.py" \
  --input_dir  "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --window     "$WINDOW" \
  --stride     "$STRIDE"
