#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TAU2_QWEN_GDN_TRITON=1
.venv/bin/python scripts/patch_vllm_qwen_mig.py
.venv/bin/python scripts/patch_vllm_qwen_gdn.py
QWEN_QUANTIZATION="${QWEN_QUANTIZATION:-none}"
QWEN_OFFLOAD="${QWEN_OFFLOAD:-prefetch}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.30}"
export QWEN_QUANTIZATION QWEN_OFFLOAD QWEN_GPU_MEMORY_UTILIZATION
extra_args=()
if [[ "$QWEN_OFFLOAD" == "prefetch" ]]; then
 extra_args+=(--offload-backend prefetch --offload-group-size 2 --offload-num-in-group 1 --offload-prefetch-step 1)
elif [[ "$QWEN_OFFLOAD" != "none" ]]; then
 echo "Unsupported QWEN_OFFLOAD: $QWEN_OFFLOAD" >&2
 exit 1
fi
if [[ "$QWEN_QUANTIZATION" != "none" ]]; then
 extra_args+=(--quantization "$QWEN_QUANTIZATION")
fi
.venv/bin/python - <<'PY'
import json,os,pathlib,time
port=os.environ.get('PORT','8004')
profile={'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','quantization':os.environ['QWEN_QUANTIZATION'],'offload':os.environ['QWEN_OFFLOAD'],'gpu_memory_utilization':float(os.environ['QWEN_GPU_MEMORY_UTILIZATION']),'device':os.environ['CUDA_VISIBLE_DEVICES'],'port':int(port),'launched_at':time.time()}
p=pathlib.Path('outputs/logs')/f'serving-{port}.json';p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(profile,indent=2));tmp.replace(p)
PY
exec .venv-vllm-qwen/bin/python -m vllm.entrypoints.openai.api_server \
 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
 --served-model-name qwen3.5-4b --host 127.0.0.1 --port "${PORT:-8004}" \
 --language-model-only --dtype bfloat16 "${extra_args[@]}" \
 --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION" --max-model-len 16384 --max-num-seqs "${QWEN_MAX_NUM_SEQS:-1}" \
 --max-num-batched-tokens "${QWEN_MAX_BATCHED_TOKENS:-528}" --enable-prefix-caching \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
 --no-enable-log-requests "$@"
