#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${OPSD_GPU:-MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04}"
CONFIG="${OPSD_CONFIG:-configs/opsd-retail-hint.json}"
COMMAND="${1:-check}"
shift || true
if [[ "$COMMAND" == run ]]; then
  RUN_DIR="${OPSD_RUN_DIR:-outputs/retail-hint/run-$(date -u +%Y%m%dT%H%M%S)-$$}"
  mkdir -p "$RUN_DIR"
  for STAGE in check collect prepare-hint train; do
    .venv/bin/python -m tau2_grpo.opsd_retail "$STAGE" --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
  done
else
  exec .venv/bin/python -m tau2_grpo.opsd_retail "$COMMAND" --config "$CONFIG" "$@"
fi
