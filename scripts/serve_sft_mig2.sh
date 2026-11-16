#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/patch_vllm_mig.py
.venv/bin/python scripts/patch_vllm_cohere_norm.py
.venv/bin/python scripts/prepare_vllm_model.py
exec .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
 --model "${MODEL_PATH:-outputs/vllm-model-525}" \
 --served-model-name tiny-aya-function-calling --host 127.0.0.1 --port "${PORT:-8002}" \
 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
 --gpu-memory-utilization 0.36 --swap-space 0 \
 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill --max-num-batched-tokens 512 "$@"
