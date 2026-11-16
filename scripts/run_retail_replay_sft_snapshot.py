"""Smoke the longest snapshot examples, then SFT from the post-Glaive-RL adapter."""
import argparse,fcntl,json,math,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/sft-retail-replay-post-glaive-rl.json');parser.add_argument('--expected-records',type=int,default=0);args=parser.parse_args()
CONFIG=Path(args.config).resolve();OUT=Path(json.loads(CONFIG.read_text())['output_dir']);OUT.mkdir(parents=True,exist_ok=True)
def state(stage,**extra):
 p=OUT/'launch-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'updated_at':time.time(),**extra},indent=2)+'\n');t.replace(p)
def run(config):
 command=['bash','scripts/run_sft.sh','--config',str(config)]
 p=subprocess.Popen(command,cwd=ROOT,env=env)
 state('sft_running',pid=p.pid,config=str(config))
 code=p.wait()
 if code:
  state('failed',exit_code=code,config=str(config));raise SystemExit(code)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
state('waiting_for_training_lock')
with (ROOT/'outputs/.retail-training.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 cfg=json.loads(CONFIG.read_text());smoke=OUT/'memory-smoke';smoke.mkdir(exist_ok=True)
 from tau2_grpo.sft_data import read_records,validate_task_partitions
 train=read_records(Path(cfg['dataset_path']));val=read_records(Path(cfg['eval_dataset_path']))
 validate_task_partitions(train,val,json.loads(Path(cfg['task_split_manifest']).read_text()))
 assert train and val
 if args.expected_records:assert len(train)+len(val)==args.expected_records
 for name,rows in [('train',train),('validation',val)]:
  r=max(rows,key=lambda r:r['context_tokens']);(smoke/(name+'.jsonl')).write_text(json.dumps(r,ensure_ascii=False)+'\n')
 smoke_cfg=dict(cfg,dataset_path=str(smoke/'train.jsonl'),eval_dataset_path=str(smoke/'validation.jsonl'),output_dir=str(OUT/'smoke-runs'),max_steps=1,gradient_accumulation_steps=1,eval_steps=1,save_steps=1,warmup_ratio=0.,wandb_run_name=cfg['wandb_run_name']+'-memory-smoke')
 (smoke/'config.json').write_text(json.dumps(smoke_cfg,indent=2)+'\n')
 print(f'Longest-example memory smoke, then full SFT: {len(train)} train, {len(val)} validation; {cfg["num_train_epochs"]} epochs; {math.ceil(len(train)/(cfg["batch_size"]*cfg["gradient_accumulation_steps"]))*cfg["num_train_epochs"]} optimizer steps',flush=True)
 run(smoke/'config.json')
 state('memory_smoke_passed',full_config=str(CONFIG))
 run(CONFIG)
 state('complete',full_config=str(CONFIG))
