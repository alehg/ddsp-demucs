#!/bin/bash
# Evaluate trained model

set -e

EXP_NAME=${1:-exp001_baseline_ddsp}
CHECKPOINT=${2:-best}

echo "Evaluating experiment: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT"

python -m ddsp_demucs.evaluate \
    --exp_name "$EXP_NAME" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "results/$EXP_NAME"

echo "Evaluation complete!"

