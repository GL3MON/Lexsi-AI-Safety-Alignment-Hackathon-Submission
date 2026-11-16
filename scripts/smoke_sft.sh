#!/usr/bin/env bash
# Three real Tiny Aya samples on the selected MIG device; W&B remains online by default.
# ./scripts/smoke_sft.sh --backend aligntune
# ./scripts/smoke_sft.sh --backend trl
set -euo pipefail

SFT_SMOKE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_SMOKE_PYTHON="$SFT_SMOKE_PROJECT/.venv/bin/python"
if [[ ! -x "$SFT_SMOKE_PYTHON" ]]; then
  echo "Missing virtual environment: $SFT_SMOKE_PYTHON" >&2
  exit 1
fi

if [[ -n "${SFT_SMOKE_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$SFT_SMOKE_GPU"
else
  CUDA_VISIBLE_DEVICES="$("$SFT_SMOKE_PYTHON" -c '
import re, subprocess, sys
devices = subprocess.check_output(["nvidia-smi", "-L"], text=True)
match = re.search(r"Device\s+0:\s+\(UUID:\s+(MIG-[^)]+)\)", devices)
if not match:
    sys.exit("MIG device 0 not found; set SFT_SMOKE_GPU explicitly")
print(match.group(1))
')"
  export CUDA_VISIBLE_DEVICES
fi
echo "SFT smoke CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Smaller optimizer states are necessary on the currently occupied MIG 2 slice.
# Full training defaults are unchanged. Later CLI overrides take precedence.
exec "$SFT_SMOKE_PROJECT/scripts/run_sft.sh" \
  --max-steps 3 \
  --set 'output_dir="outputs/sft-smoke"' \
  --set max_train_samples=3 --set max_eval_samples=1 \
  --set batch_size=1 --set gradient_accumulation_steps=1 \
  --set warmup_ratio=0 --set eval_steps=3 --set save_steps=3 \
  --set activation_offloading=true --set 'optim="adamw_bnb_8bit"' \
  --set log_sample_count=1 --set sample_max_new_tokens=96 \
  "$@"
