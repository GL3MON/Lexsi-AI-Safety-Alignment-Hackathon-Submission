#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/retail-data/retail-sft-corrective-v2/progress.json')
r=json.loads(p.read_text())
if not r.get('generation_complete') or not r.get('complete'):
    raise SystemExit('Collection has missing generations or selected examples; inspect progress.json before finalizing.')
PY
exec .venv/bin/python scripts/finalize_retail_sft.py \
  --input-dir outputs/retail-data/retail-sft-corrective-v2 \
  --output outputs/retail-data/retail-sft-corrective-v2-final \
  --endpoint http://127.0.0.1:8004/v1 "$@"
