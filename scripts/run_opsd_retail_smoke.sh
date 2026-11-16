#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${OPSD_GPU:-MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04}"
CONFIG="${OPSD_SMOKE_CONFIG:-configs/opsd-retail-smoke.json}"
RUN_DIR="${OPSD_SMOKE_RUN_DIR:-outputs/opsd-retail-smoke/run-$(date -u +%Y%m%dT%H%M%S)-$$}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$GPU" == "MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04" ]] && pgrep -af '[h]old_mig0_memory.py' >/dev/null; then
  echo "MIG0 memory holder is still running; stop it before the smoke." >&2
  exit 2
fi
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Smoke run directory is not empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
echo "OPSD smoke artifacts: $RUN_DIR"

.venv/bin/python -m tau2_grpo.opsd_retail check --config "$CONFIG" --task-ids 59,113
.venv/bin/python -m tau2_grpo.opsd_retail evaluate --config "$CONFIG" --run-dir "$RUN_DIR" \
  --eval-stage before --task-ids 69
.venv/bin/python -m tau2_grpo.opsd_retail collect --config "$CONFIG" --run-dir "$RUN_DIR" \
  --partition rl --task-ids 59,113
.venv/bin/python -m tau2_grpo.opsd_retail diagnose --config "$CONFIG" --run-dir "$RUN_DIR"
.venv/bin/python -m tau2_grpo.opsd_retail validate --config "$CONFIG" --run-dir "$RUN_DIR"

if [[ -s "$RUN_DIR/verified_recovery.jsonl" ]]; then
  .venv/bin/python -m tau2_grpo.opsd_retail train --config "$CONFIG" --run-dir "$RUN_DIR"
  .venv/bin/python -m tau2_grpo.opsd_retail evaluate --config "$CONFIG" --run-dir "$RUN_DIR" \
    --eval-stage after --checkpoint latest --task-ids 69
else
  echo "No verified hint-improved recovery was found; OPSD optimizer and after-evaluation are skipped."
fi
