#!/usr/bin/env bash
# Prepare the pinned Hermes single/multi-turn mix and disjoint SFT/RL pools.
# Existing output directories are never overwritten; choose --output for a rebuild.
# Examples:
#   ./scripts/prepare_sft_data.sh
#   ./scripts/prepare_sft_data.sh --output outputs/function-calling-data-v2
#   ./scripts/prepare_sft_data.sh --seed 42 --rl-fraction 0.2 --max-tokens 8192
set -euo pipefail

SFT_DATA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_DATA_PYTHON="$SFT_DATA_PROJECT/.venv/bin/python"

if [[ ! -x "$SFT_DATA_PYTHON" ]]; then
  echo "Missing virtual environment: $SFT_DATA_PYTHON. Run bash $SFT_DATA_PROJECT/scripts/setup.sh first." >&2
  exit 1
fi

exec "$SFT_DATA_PYTHON" -m tau2_grpo.prepare_function_calling \
  --output "$SFT_DATA_PROJECT/outputs/function-calling-data" "$@"
