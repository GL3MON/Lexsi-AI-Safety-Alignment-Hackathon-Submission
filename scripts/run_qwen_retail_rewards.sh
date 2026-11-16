#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
QWEN_TOKENIZER="$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-4B",revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",local_files_only=True,allow_patterns=["*.json","*.jinja","*.txt","*.model","*.tiktoken"]))')"
exec .venv/bin/python scripts/analyze_tau2_rewards.py \
 --api-mode chat --base-url http://127.0.0.1:8004/v1 --model qwen3.5-4b \
 --tokenizer "$QWEN_TOKENIZER" --context-limit 16384 --max-new-tokens 512 \
 --tasks 2 --samples-per-task 2 --max-actions 24 \
 --split-manifest outputs/retail-data/split_manifest.json --partition sft \
 --task-ids 88,2 --wandb-project tau2-retail "$@"
