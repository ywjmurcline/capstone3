#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_svm.sh  —  Train multi-modal SVM
#
# Two modes:
#   SINGLE   one participant directory  (set MODE=single)
#   MULTI    multiple participants      (set MODE=multi)
# ─────────────────────────────────────────────────────────────────────────────

PYTHON=/Users/lily/Documents/myApps/Capstone_Saveme/venv-psychopy/bin/python
SCRIPT=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/svm.py

# ── Mode: single | multi ─────────────────────────────────────────────────────
MODE=multi

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODE
# ─────────────────────────────────────────────────────────────────────────────
BASE=/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/mym
WINDOWS_DIR="$BASE/windows"
TRIALS_JSON="$BASE/trials.json"

# ─────────────────────────────────────────────────────────────────────────────
# MULTI MODE
# ─────────────────────────────────────────────────────────────────────────────
PARTICIPANTS_JSON=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/participants.json

TRAIN_PARTICIPANTS='["mym","ywj"]'

# JSON list of test configs.  Example: '["mym", "abc", ["mym","abc"]]'
TEST_PARTICIPANTS='["mym","ywj"]'

TEST_SPLIT=0.2

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/Users/lily/Documents/myApps/Capstone_Saveme/17_new_model/work_dir/svm/$(date +%Y-%m-%d_%H-%M-%S)

# Ground truth type: valence | arousal | va | emotion
GT_TYPE=arousal

EMOTION_CLASSES="neutral,amusing,fear"

MODALITIES="ppg,ear"

PPG_VARIANT=finger_pre
EAR_VARIANT=ear
US_VARIANT=nodiff

VAL_SPLIT=0.2
CV_FOLDS=5

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
        --val_split         $VAL_SPLIT \
        --cv_folds          $CV_FOLDS

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
        --val_split          $VAL_SPLIT \
        --cv_folds           $CV_FOLDS \
        --test_split         $TEST_SPLIT

else
    echo "Unknown MODE='$MODE'. Set MODE=single or MODE=multi."
    exit 1
fi
