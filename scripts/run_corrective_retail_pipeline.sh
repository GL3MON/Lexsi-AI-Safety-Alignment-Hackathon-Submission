#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/generate_corrective_retail_sft.sh "$@"
if [[ -d outputs/retail-data/retail-sft-corrective-v2-final ]]; then
  scripts/prepare_corrective_retail_sft.sh --resume
else
  scripts/prepare_corrective_retail_sft.sh
fi
