#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TAU2_CHECKOUT="${TAU2_CHECKOUT:-$PROJECT/../tau2-bench}"
export UV_FROZEN=1
if [[ ! -x "$PROJECT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$PROJECT/.venv"
fi
uv pip install --python "$PROJECT/.venv/bin/python" -e "$TAU2_CHECKOUT[gym]" -e "$PROJECT" -r "$PROJECT/requirements-gpu.txt" \
  --index "https://download.pytorch.org/whl/cu126" \
  --index-strategy unsafe-best-match \
  "torch==2.11.0+cu126" "torchvision==0.26.0+cu126"
"$PROJECT/.venv/bin/tau2-grpo" doctor --tau2 "$TAU2_CHECKOUT"
