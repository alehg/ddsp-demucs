#!/usr/bin/env bash
# Train residual DDSP (same data path / flags style as train_exp001_ddsp_direct.py).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/exp001_dualfir_tuned.yaml}"
EXP_NAME="${EXP_NAME:-exp001_ddsp_residual}"

python scripts/train_exp001_ddsp_residual.py \
  --exp-name "$EXP_NAME" \
  --config "$CONFIG" \
  --seed 1337 \
  "$@"
