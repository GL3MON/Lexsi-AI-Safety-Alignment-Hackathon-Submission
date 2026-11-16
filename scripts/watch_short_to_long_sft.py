#!/usr/bin/env python3
"""Wait for verified short SFT + long-data completion, then launch stage two once."""
import argparse,fcntl,json,math,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def best_adapter(run):
 state=read(run/'trainer_state.json');best=Path(state['best_model_checkpoint'])
 if not best.is_absolute():best=ROOT/best
 best=best.resolve()
 if best.parent!=run.resolve():raise ValueError('Best checkpoint is outside the expected short run')
 for name in ('adapter_config.json','adapter_model.safetensors'):
  if not (best/name).is_file() or not (best/name).stat().st_size:raise ValueError('Missing best adapter file: '+str(best/name))
 metric=state.get('best_metric')
 if not isinstance(metric,(int,float)) or not math.isfinite(metric):raise ValueError('Invalid best validation loss')
 return best,state

def manifest_for(data):
 train=[json.loads(s) for s in (data/'sft.jsonl').read_text().splitlines() if s.strip()]
 val=[json.loads(s) for s in (data/'validation.jsonl').read_text().splitlines() if s.strip()]
 if (len(train),len(val))!=(450,50):raise ValueError('Expected 450 long training and 50 validation samples')
 groups=[r['group_id'] for r in train+val]
 if len(set(groups))!=500:raise ValueError('Duplicate or overlapping long scenario families')
 if any(not all(r.get('checks',{}).values()) or not r.get('checks') for r in train+val):raise ValueError('Missing/failed long-data execution checks')
 return {'domain':'retail','split_unit':'scenario_family','partitions':{'sft':[r['task_id'] for r in train],'validation':[r['task_id'] for r in val],'rl':[],'test':[],'quarantine':[]}}

def main(args):
 os.chdir(ROOT);folder=ROOT/'outputs/watchers/short-to-long-sft';folder.mkdir(parents=True,exist_ok=True)
 lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statefile=folder/'state.json'
 if statefile.exists() and read(statefile).get('status') in ('launching','training','complete','training_failed'):
  raise RuntimeError('Stage two was already launched; inspect watcher state before any new launch')
 short=Path(args.short_run);data=Path(args.long_data)
 trainer=read('outputs/logs/sft-retail-short-2k-from-hermes-process.json')['pid']
 generator=read('outputs/logs/long-clarification-500-mig2-process.json')['pid']
 while True:
  short_ready=all((short/f).is_file() for f in ('metrics.json','trainer_state.json','generated_samples.json'))
  long_ready=(data/'report.json').is_file() and read(data/'report.json').get('complete') is True
  running={'short_training':alive(trainer),'long_generation':alive(generator)}
  status={'status':'waiting','short_run':str(short),'long_data':str(data),'short_artifacts_ready':short_ready,'long_data_ready':long_ready,'running':running,'long_accepted':len(list((data/'records').glob('*.json'))),'updated_at':time.time()}
  atomic(statefile,status)
  print(json.dumps(status),flush=True)
  if not running['short_training'] and not short_ready:raise RuntimeError('Short SFT exited without complete artifacts; stage two not launched')
  if not running['long_generation'] and not long_ready:raise RuntimeError('Long generation exited without a complete report; stage two not launched')
  if short_ready and long_ready and not any(running.values()):break
  time.sleep(args.interval)
 metrics=read(short/'metrics.json');cfg=read(short/'config.json')
 if metrics.get('epoch',0)<cfg['num_train_epochs'] or metrics.get('changed_adapter_tensors',0)<=0:raise ValueError('Short training did not complete its scheduled epochs with adapter updates')
 for key in ('train_loss','eval_loss'):
  if not isinstance(metrics.get(key),(float,int)) or not math.isfinite(metrics[key]):raise ValueError('Invalid short metrics: '+key)
 log=Path('outputs/logs/sft-retail-short-2k-from-hermes.log').read_text()
 if 'Saved adapter:' not in log or 'Traceback (most recent call last)' in log:raise ValueError('Short run log does not indicate clean completion')
 best,state=best_adapter(short)
 report=read(data/'report.json')
 if report.get('samples')!=500:raise ValueError('Long report must contain 500 samples')
 manifest=data/'task-split-manifest.json';atomic(manifest,manifest_for(data))
 cfg.update(backend='aligntune',train_order='tool_call_count',initial_adapter=str(best),dataset_path=str(data.resolve()/'sft.jsonl'),eval_dataset_path=str(data.resolve()/'validation.jsonl'),task_split_manifest=str(manifest.resolve()),rl_holdout_path=None,output_dir='outputs/sft-retail-long-500-from-short-best',num_train_epochs=2,max_steps=-1,resume_from_checkpoint=None,evaluate_before_training=True,eval_steps=25,save_steps=25,wandb_mode='online',wandb_run_name='retail-long-500-from-short-best-2epochs',wandb_tags=['sft','retail','long-500','from-short-2k-best','preambles'])
 config_path=ROOT/'configs/sft-retail-long-500-from-short-best.json';atomic(config_path,cfg)
 status.update(status='launching',best_checkpoint=str(best),best_validation_loss=state['best_metric'],config=str(config_path),updated_at=time.time());atomic(statefile,status)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
 logfile=ROOT/'outputs/logs/sft-retail-long-500-from-short-best.log'
 with logfile.open('w') as f:
  proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','tau2_grpo.sft','--config',str(config_path)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 status.update(status='training',pid=proc.pid,log=str(logfile),updated_at=time.time());atomic(statefile,status)
 print('LAUNCHED '+json.dumps(status),flush=True)
 code=proc.wait();status.update(status='complete' if code==0 else 'training_failed',exit_code=code,updated_at=time.time());atomic(statefile,status)
 if code:raise RuntimeError('Long SFT failed with exit code '+str(code))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--short-run',default='outputs/sft-retail-short-2k-from-hermes/run-5c6ee49620');p.add_argument('--long-data',default='outputs/retail-data/long-clarification-500-mig2');p.add_argument('--interval',type=float,default=15);args=p.parse_args()
 try:main(args)
 except Exception as exc:
  error=ROOT/'outputs/watchers/short-to-long-sft/error.json';error.parent.mkdir(parents=True,exist_ok=True);atomic(error,{'error':repr(exc),'time':time.time()});raise
