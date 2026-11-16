#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${RETAIL_RL_GPU:-MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04}"
if [[ "${1:-check}" != "check" ]]; then
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/logs/retail-sft-training-process.json')
if p.exists():
    pid=json.loads(p.read_text())['pid']
    cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and b'tau2_grpo.sft' in cmd.read_bytes():
        raise SystemExit('SFT is still running. Allocate another GPU or wait before launching RL.')
PY
fi
exec .venv/bin/python -m tau2_grpo.rl "${@:-check}"
