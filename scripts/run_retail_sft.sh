#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${RETAIL_SFT_GPU:-MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04}"
export SFT_CONFIG=configs/sft-retail.json
exec scripts/run_sft.sh "$@"
