"""Run a five-error smoke gate, then 500 recovery and 1500 clean samples. No training."""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/retail-data/retail-replay-2000'
OUT.mkdir(parents=True,exist_ok=True)
def state(stage,**kw):
    (OUT/'pipeline-status.json').write_text(json.dumps({'stage':stage,'updated_at':time.time(),**kw},indent=2)+'\n')
def run(args):
    command=[sys.executable,'-u','scripts/generate_retail_replay_sft.py','--output',str(OUT)]+args
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:
        state('failed',exit_code=result.returncode,command=command);raise SystemExit(result.returncode)
state('smoke',recovery_target=500,clean_target=1500)
run(['--recovery-count','5','--clean-count','0'])
records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))]
assert len(records)>=5 and {r['masked_faults'][0]['kind'] for r in records[:5]}=={'json_structure','unknown_tool','argument_schema','wrong_identifier','wrong_operation'}
assert all(r['context_tokens']<=8192 and r['semantic_review']['accept'] for r in records)
state('generation',recovery_target=500,clean_target=1500,smoke_passed=True)
run(['--recovery-count','500','--clean-count','1500'])
from tau2_grpo.sft_data import read_records,validate_task_partitions
train=read_records(OUT/'sft.jsonl');val=read_records(OUT/'validation.jsonl')
validate_task_partitions(train,val,json.loads((OUT/'task-split-manifest.json').read_text()))
assert len(train)+len(val)==2000
state('complete',train=len(train),validation=len(val),training_launched=False)
