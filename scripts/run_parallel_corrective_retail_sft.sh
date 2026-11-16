#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/run_parallel_retail_sft.py \
 --source outputs/retail-data/retail-sft-corrective-v2 \
 --source outputs/retail-data/retail-sft-corrective-v2-mig0 \
 --output outputs/retail-data/retail-sft-corrective-parallel-v2 \
 --corrective-prompts --collection-mode all-generations --manage-servers "$@"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/retail-data/retail-sft-corrective-parallel-v2/completed.json')
if not p.exists() or not json.loads(p.read_text()).get('generation_complete'):
    raise SystemExit('Collection is incomplete; final preparation has not started.')
PY
extra=()
if [[ -d outputs/retail-data/retail-sft-corrective-v2-final ]]; then extra+=(--resume); fi
exec .venv/bin/python scripts/finalize_retail_sft.py \
 --input-dir outputs/retail-data/retail-sft-corrective-parallel-v2 \
 --output outputs/retail-data/retail-sft-corrective-v2-final \
 --endpoint http://127.0.0.1:8004/v1 --endpoint http://127.0.0.1:8005/v1 "${extra[@]}"
