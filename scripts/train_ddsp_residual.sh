#!/bin/bash
# Train DDSP residual model

set -e

CONFIG=${1:-configs/base.yaml}
EXP_NAME=${2:-exp001_baseline_ddsp}

echo "Training with config: $CONFIG"
echo "Experiment name: $EXP_NAME"

python -m ddsp_demucs.train \
    --config "$CONFIG" \
    --exp_name "$EXP_NAME" \
    --gpu 0

echo "Training complete!"

