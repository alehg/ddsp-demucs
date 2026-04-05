#!/bin/bash
# Train Experiment 001 (DDSP direct) with local pipeline defaults.

set -euo pipefail

CONFIG=${1:-configs/base.yaml}
ENV_CONFIG=${2:-env/config.yaml}
EXP_NAME=${3:-exp001_ddsp_direct}

echo "Config: $CONFIG"
echo "Env config: $ENV_CONFIG"
echo "Experiment: $EXP_NAME"

python scripts/train_exp001_ddsp_direct.py \
  --config "$CONFIG" \
  --env-config "$ENV_CONFIG" \
  --exp-name "$EXP_NAME"

echo "Training run finished."

