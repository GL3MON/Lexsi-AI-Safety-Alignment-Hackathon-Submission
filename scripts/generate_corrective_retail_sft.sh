#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec scripts/generate_retail_sft.sh \
  --output outputs/retail-data/retail-sft-corrective-v2 \
  --base-url http://127.0.0.1:8004/v1 --model qwen3.5-4b \
  --agent-prompt-file templates/retail_sft_agent_corrective.txt \
  --user-prompt-file templates/retail_sft_customer_corrective.txt \
  --customer-profiles identity-clarification,lookup-recovery,confirmation-execution \
  --sft-per-task 3 --validation-per-task 1 --collection-mode all-generations \
  --max-actions 40 --seed 20261110 "$@"
