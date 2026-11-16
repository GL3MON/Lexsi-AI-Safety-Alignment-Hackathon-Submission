#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/generate_long_retail_sft.py "$@"
