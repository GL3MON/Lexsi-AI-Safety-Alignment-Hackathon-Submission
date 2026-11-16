#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${RETAIL_USER_GPU:-MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b}"
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/patch_vllm_mig.py
.venv/bin/python scripts/patch_vllm_cohere_norm.py
exec .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
 --model CohereLabs/tiny-aya-global \
 --revision af89d219b53ed9b13b8a4645f9c8028973510324 \
 --served-model-name tiny-aya-base-user --host 127.0.0.1 --port "${PORT:-8003}" \
 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
 --gpu-memory-utilization 0.36 --swap-space 0 \
 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill \
 --max-num-batched-tokens 512 "$@"
