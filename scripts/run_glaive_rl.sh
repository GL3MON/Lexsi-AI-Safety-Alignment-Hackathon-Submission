#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m tau2_grpo.glaive_rl "${@:-check}"
