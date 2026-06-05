#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cnn.sh  —  Train multi-modal CNN (22_models)
#
# Split strategy: per-participant, per-class shuffle → 60-20-20 (train-val-test)
#   then global shuffle within each split. No undersampling.
#
# Two modes:
#   SINGLE   one participant directory  (set MODE=single)
#   MULTI    multiple participants      (set MODE=multi)
# ─────────────────────────────────────────────────────────────────────────────

PYTHON=/Users/lily/Documents/myApps/Capstone_Saveme/venv-psychopy/bin/python
SCRIPT=/Users/lily/Documents/myApps/Capstone_Saveme/22_models/cnn.py

# ── Mode: single | multi ─────────────────────────────────────────────────────
MODE=single

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODE — one participant
# ─────────────────────────────────────────────────────────────────────────────
BASE=/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/qishui
WINDOWS_DIR="$BASE/windows_10_5"
TRIALS_JSON="$BASE/trials.json"
PARTICIPANT_ID="qishui"

# ─────────────────────────────────────────────────────────────────────────────
# MULTI MODE — multiple participants
# participants.json: [{"participant_id": "mym", "directory": "/path/to/mym"}, ...]
# Each participant's data is split 60-20-20 per class independently.
# ─────────────────────────────────────────────────────────────────────────────
PARTICIPANTS_JSON=/Users/lily/Documents/myApps/Capstone_Saveme/22_models/participants.json

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/Users/lily/Documents/myApps/Capstone_Saveme/22_models/work_dir/cnn/$(date +%Y-%m-%d_%H-%M-%S)

# Ground truth type: valence | arousal | va | emotion
GT_TYPE=emotion

# Required only when GT_TYPE=emotion — comma-separated emotion class names
EMOTION_CLASSES="neutral,amusing,fear,happy,disgust,sad"

# Modalities: comma-separated subset of ppg, ear, ultrasound
MODALITIES="ppg"

# Signal variants
#   PPG:  finger_pre | glass_pre | finger | glass
#   EAR:  ear | blink
#   US:   nodiff | diff
PPG_VARIANT=glass_pre
EAR_VARIANT=ear
US_VARIANT=nodiff

# Hyperparameters
EPOCHS=50
BATCH=32
LR=0.001
VAL_SPLIT=0.2
TEST_SPLIT=0.2
SEED=42

mkdir -p "$WORK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "single" ]; then
    $PYTHON "$SCRIPT" \
        "$WINDOWS_DIR" \
        --trials_json       "$TRIALS_JSON" \
        --participant_id    "$PARTICIPANT_ID" \
        --gt_type           "$GT_TYPE" \
        --emotion_classes   "$EMOTION_CLASSES" \
        --ppg_variant       "$PPG_VARIANT" \
        --ear_variant       "$EAR_VARIANT" \
        --us_variant        "$US_VARIANT" \
        --modalities        "$MODALITIES" \
        --work_dir          "$WORK_DIR" \
        --epochs            $EPOCHS \
        --batch             $BATCH \
        --lr                $LR \
        --val_split         $VAL_SPLIT \
        --test_split        $TEST_SPLIT \
        --seed              $SEED

elif [ "$MODE" = "multi" ]; then
    $PYTHON "$SCRIPT" \
        --participants_json  "$PARTICIPANTS_JSON" \
        --gt_type            "$GT_TYPE" \
        --emotion_classes    "$EMOTION_CLASSES" \
        --ppg_variant        "$PPG_VARIANT" \
        --ear_variant        "$EAR_VARIANT" \
        --us_variant         "$US_VARIANT" \
        --modalities         "$MODALITIES" \
        --work_dir           "$WORK_DIR" \
        --epochs             $EPOCHS \
        --batch              $BATCH \
        --lr                 $LR \
        --val_split          $VAL_SPLIT \
        --test_split         $TEST_SPLIT \
        --seed               $SEED

else
    echo "Unknown MODE='$MODE'. Set MODE=single or MODE=multi."
    exit 1
fi
