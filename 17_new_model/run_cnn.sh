#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cnn.sh  —  Train multi-modal CNN
#
# Two modes:
#   SINGLE   one participant directory  (set MODE=single)
#   MULTI    multiple participants      (set MODE=multi)
# ─────────────────────────────────────────────────────────────────────────────

PYTHON=/Users/lily/Documents/myApps/Capstone_Saveme/venv-psychopy/bin/python
SCRIPT=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/cnn.py

# ── Mode: single | multi ─────────────────────────────────────────────────────
MODE=single

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODE
# ─────────────────────────────────────────────────────────────────────────────
BASE=/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/qishui
WINDOWS_DIR="$BASE/windows"
TRIALS_JSON="$BASE/trials.json"

# ─────────────────────────────────────────────────────────────────────────────
# MULTI MODE
# participants_config: [{"participant_id": "mym", "directory": "/path/to/mym"}, ...]
# ─────────────────────────────────────────────────────────────────────────────
PARTICIPANTS_JSON=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/participants.json

# JSON list of participant IDs for training, e.g. '["mym","ywj"]'
TRAIN_PARTICIPANTS='["qishui"]'

# Test set definitions — JSON list; each element is one test evaluation.
# A bare ID tests that participant alone; a JSON array combines multiple.
# Example: '["mym", "abc", ["mym","abc"]]'
TEST_PARTICIPANTS='["qishui"]'

# Fraction of each participant's data reserved for the test set
TEST_SPLIT=0.2

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/work_dir/cnn/$(date +%Y-%m-%d_%H-%M-%S)

# Ground truth type: valence | arousal | va | emotion
GT_TYPE=emotion

# Required only when GT_TYPE=emotion — comma-separated emotion class names
# Available tags: amusing, anger, disgust, fear, happy, neutral, sad
EMOTION_CLASSES="neutral,amusing,fear,happy,disgust,sad"

# Modalities: comma-separated subset of ppg, ear, ultrasound
MODALITIES="ppg"

# Signal variants
#   PPG:  finger_pre | glass_pre | finger | glass
#   EAR:  ear (Eye Aspect Ratio, continuous) | blink (binary)
#   US:   nodiff | diff (frame-differenced)
PPG_VARIANT=finger_pre
EAR_VARIANT=ear
US_VARIANT=nodiff

# Hyperparameters
EPOCHS=50
BATCH=32
LR=0.001
VAL_SPLIT=0.2

mkdir -p "$WORK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "single" ]; then
    $PYTHON "$SCRIPT" \
        "$WINDOWS_DIR" \
        --trials_json       "$TRIALS_JSON" \
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
        --val_split         $VAL_SPLIT

elif [ "$MODE" = "multi" ]; then
    $PYTHON "$SCRIPT" \
        --participants_json  "$PARTICIPANTS_JSON" \
        --train_participants "$TRAIN_PARTICIPANTS" \
        --test_participants  "$TEST_PARTICIPANTS" \
        --test_split         $TEST_SPLIT \
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
        --test_split         $TEST_SPLIT

else
    echo "Unknown MODE='$MODE'. Set MODE=single or MODE=multi."
    exit 1
fi
