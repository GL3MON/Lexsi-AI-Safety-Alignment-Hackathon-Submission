#!/usr/bin/env bash
# Run from any directory. All arguments are forwarded to the SFT CLI.
# Examples:
#   ./scripts/run_sft.sh
#   ./scripts/run_sft.sh --prepare-only
#   ./scripts/run_sft.sh --max-steps 2 --set warmup_ratio=0
#   ./scripts/run_sft.sh --config /path/to/custom.json
#   ./scripts/run_sft.sh --backend aligntune
#   ./scripts/run_sft.sh --backend trl
set -euo pipefail

SFT_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_PYTHON="$SFT_PROJECT/.venv/bin/python"
SFT_CONFIG="${SFT_CONFIG:-$SFT_PROJECT/configs/sft.json}"

if [[ ! -x "$SFT_PYTHON" ]]; then
  echo "Missing virtual environment: $SFT_PYTHON. Run bash $SFT_PROJECT/scripts/setup.sh first." >&2
  exit 1
fi

# exec preserves exit codes and sends Ctrl-C/SIGTERM directly to the trainer.
# W&B online logging is the default in the config; the trainer checks initialization.
exec "$SFT_PYTHON" -m tau2_grpo.sft --config "$SFT_CONFIG" "$@"
