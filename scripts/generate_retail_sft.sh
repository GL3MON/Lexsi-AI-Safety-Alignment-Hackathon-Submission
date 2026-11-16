#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
QWEN_TOKENIZER="$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-4B",revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",local_files_only=True,allow_patterns=["*.json","*.jinja","*.txt","*.model","*.tiktoken"]))')"
exec .venv/bin/python scripts/generate_retail_sft.py \
 --teacher-tokenizer "$QWEN_TOKENIZER" "$@"
