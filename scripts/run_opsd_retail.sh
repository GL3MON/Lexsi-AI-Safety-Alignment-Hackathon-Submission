#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

COMMAND="${1:-check}"
shift || true
CONFIG="${OPSD_CONFIG:-configs/opsd-retail.json}"
GPU="${OPSD_GPU:-MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$GPU" == "MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04" && "$COMMAND" != "check" && "$COMMAND" != "diagnose" ]]; then
  if pgrep -af '[h]old_mig0_memory.py' >/dev/null; then
    echo "MIG0 is intentionally reserved by the memory holder. Stop that holder before GPU stages." >&2
    exit 2
  fi
fi

if [[ "$COMMAND" == "run" ]]; then
  RUN_DIR="${OPSD_RUN_DIR:-outputs/opsd-retail/run-$(date -u +%Y%m%dT%H%M%S)-$$}"
  if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Run directory is not empty: $RUN_DIR" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR"
  echo "OPSD run directory: $RUN_DIR"
  .venv/bin/python -m tau2_grpo.opsd_retail check --config "$CONFIG" "$@"
  echo "=== OPSD baseline: hint-free validation ==="
  .venv/bin/python -m tau2_grpo.opsd_retail evaluate --eval-stage before --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
  for STAGE in collect diagnose validate train; do
    echo "=== OPSD stage: $STAGE ==="
    .venv/bin/python -m tau2_grpo.opsd_retail "$STAGE" --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
  done
  echo "=== OPSD final: hint-free validation ==="
  .venv/bin/python -m tau2_grpo.opsd_retail evaluate --eval-stage after --checkpoint latest --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
  exit 0
fi

case "$COMMAND" in
  check|collect|diagnose|validate|train|evaluate) ;;
  *) echo "Usage: $0 {check|collect|diagnose|validate|train|evaluate|run} [pipeline arguments]" >&2; exit 2 ;;
esac
exec .venv/bin/python -m tau2_grpo.opsd_retail "$COMMAND" --config "$CONFIG" "$@"
