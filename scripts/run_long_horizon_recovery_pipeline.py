"""Generate 500 verified conversations, smoke SFT, then train under the GPU lock."""
import fcntl,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'outputs/retail-data/long-horizon-recovery-500';r.mkdir(parents=True,exist_ok=True)
state=r/'pipeline-status.json'
def status(stage,**extra):state.write_text(json.dumps({'stage':stage,**extra},indent=2)+'\n')
def run(command):
 result=subprocess.run(command,cwd=ROOT,env=env)
 if result.returncode:
  status('failed',command=command,exit_code=result.returncode)
  raise SystemExit(result.returncode)
env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04')
status('generating',target_conversations=500)
run(['.venv/bin/python','-u','scripts/generate_long_horizon_recovery.py','--count','500','--output',str(r),'--endpoint','http://127.0.0.1:8004/v1','--model','qwen3.5-4b'])
report=json.loads((r/'report.json').read_text());assert report['complete'] and report['completed']==500 and report['max_window_tokens']<=8192
from tau2_grpo.sft_data import read_records,validate_task_partitions
train=read_records(r/'sft.jsonl');val=read_records(r/'validation.jsonl');validate_task_partitions(train,val,json.loads((r/'task-split-manifest.json').read_text()))
status('waiting_for_gpu',**report)
with (ROOT/'outputs/.retail-training.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 cfg=json.loads((ROOT/'configs/sft-retail-long-horizon-recovery-500.json').read_text())
 smoke=r/'sft-smoke';smoke.mkdir(exist_ok=True)
 for name,rows in [('train',train),('validation',val)]:
  record=max(rows,key=lambda x:x['context_tokens']);(smoke/(name+'.jsonl')).write_text(json.dumps(record,ensure_ascii=False)+'\n')
 cfg.update(dataset_path=str(smoke/'train.jsonl'),eval_dataset_path=str(smoke/'validation.jsonl'),max_steps=1,
  gradient_accumulation_steps=1,eval_steps=1,save_steps=1,warmup_ratio=0.,output_dir=str(ROOT/'outputs/sft-retail-long-horizon-recovery-smoke'),wandb_run_name='retail-long-horizon-recovery-memory-smoke')
 (smoke/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 status('sft_memory_smoke',**report)
 run(['bash','scripts/run_sft.sh','--config',str(smoke/'config.json')])
 status('sft_training',**report)
 run(['bash','scripts/run_sft.sh','--config','configs/sft-retail-long-horizon-recovery-500.json'])
 status('complete',**report)
