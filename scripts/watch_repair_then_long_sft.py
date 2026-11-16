"""Automatic short-best -> exact-copy repair -> long curriculum pipeline."""
import fcntl,json,math,os,subprocess,time
from pathlib import Path
from scripts.watch_short_to_long_sft import ROOT,read,atomic,alive,best_adapter,manifest_for

FOLDER=ROOT/'outputs/watchers/repair-then-long-sft'

def main():
 os.chdir(ROOT);FOLDER.mkdir(parents=True,exist_ok=True)
 lock=(FOLDER/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sf=FOLDER/'state.json'
 if sf.exists() and read(sf).get('status') in ('repair_training','long_training','complete','failed'):
  raise RuntimeError('Pipeline already launched; inspect state before restart')
 short=ROOT/'outputs/sft-retail-short-2k-from-hermes/run-5c6ee49620'
 repair=ROOT/'outputs/retail-data/copy-repair-500';long=ROOT/'outputs/retail-data/long-clarification-500-mig2'
 shortpid=read(ROOT/'outputs/logs/sft-retail-short-2k-from-hermes-process.json')['pid']
 genpid=read(ROOT/'outputs/logs/copy-repair-500-process.json')['pid']
 def update(status,**kwargs):
  payload={'status':status,'updated_at':time.time(),**kwargs};atomic(sf,payload);print(json.dumps(payload),flush=True)
 while True:
  short_done=all((short/n).is_file() for n in ('metrics.json','trainer_state.json','generated_samples.json')) and not alive(shortpid)
  repair_done=(repair/'report.json').exists() and read(repair/'report.json').get('complete') is True and not alive(genpid)
  long_done=(long/'report.json').exists() and read(long/'report.json').get('complete') is True
  update('waiting',short_complete=short_done,repair_data_complete=repair_done,long_data_complete=long_done,repair_accepted=len(list((repair/'records').glob('*.json'))))
  if not alive(shortpid) and not short_done:raise RuntimeError('Short SFT exited without complete artifacts')
  if not alive(genpid) and not repair_done:raise RuntimeError('Repair generation exited without complete report')
  if short_done and repair_done and long_done:break
  time.sleep(15)
 short_metrics=read(short/'metrics.json')
 if short_metrics.get('epoch',0)<2 or short_metrics.get('changed_adapter_tensors',0)<=0:raise ValueError('Short run did not complete both epochs with adapter updates')
 shortlog=(ROOT/'outputs/logs/sft-retail-short-2k-from-hermes.log').read_text()
 if 'Saved adapter:' not in shortlog or 'Traceback (most recent call last)' in shortlog:raise ValueError('Short run did not finish cleanly')
 best,shortstate=best_adapter(short)
 basecfg=read(short/'config.json')
 def stage(label,data,initial,rate,order,output):
  manifest=data/'task-split-manifest.json';atomic(manifest,manifest_for(data))
  cfg=dict(basecfg);cfg.update(backend='aligntune',initial_adapter=str(initial),dataset_path=str(data/'sft.jsonl'),eval_dataset_path=str(data/'validation.jsonl'),task_split_manifest=str(manifest),rl_holdout_path=None,output_dir=str(output),num_train_epochs=2,max_steps=-1,resume_from_checkpoint=None,evaluate_before_training=True,learning_rate=rate,train_order=order,eval_steps=25,save_steps=25,wandb_mode='online',wandb_run_name='retail-'+label+'-2epochs',wandb_tags=['sft','retail',label,'preambles'])
  cp=ROOT/'configs'/('sft-retail-'+label+'.json');atomic(cp,cfg)
  logfile=ROOT/'outputs/logs'/('sft-retail-'+label+'.log')
  if output.exists() and list(output.glob('run-*')):raise RuntimeError('Refusing duplicate stage output '+str(output))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
  with logfile.open('w') as f:proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','tau2_grpo.sft','--config',str(cp)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  update('repair_training' if label=='copy-repair-500' else 'long_training',pid=proc.pid,config=str(cp),log=str(logfile),initial_adapter=str(initial),learning_rate=rate,train_order=order)
  code=proc.wait()
  if code:raise RuntimeError(label+' exited with '+str(code))
  runs=list(output.glob('run-*'));assert len(runs)==1
  run=runs[0];m=read(run/'metrics.json')
  if m.get('epoch',0)<2 or m.get('changed_adapter_tensors',0)<=0:raise RuntimeError('Incomplete stage metrics')
  if not all(math.isfinite(m[k]) for k in ('train_loss','eval_loss')):raise RuntimeError('Nonfinite stage loss')
  return run
 repair_run=stage('copy-repair-500',repair,best,2e-5,'random',ROOT/'outputs/sft-retail-copy-repair-500')
 repair_best,repair_state=best_adapter(repair_run)
 atomic(FOLDER/'repair-completed.json',{'short_best':str(best),'repair_run':str(repair_run),'repair_best':str(repair_best),'repair_best_validation_loss':repair_state['best_metric']})
 long_run=stage('long-500-after-copy-repair',long,repair_best,basecfg['learning_rate'],'tool_call_count',ROOT/'outputs/sft-retail-long-500-after-copy-repair')
 update('complete',repair_run=str(repair_run),repair_best=str(repair_best),long_run=str(long_run))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  FOLDER.mkdir(parents=True,exist_ok=True);atomic(FOLDER/'error.json',{'error':repr(exc),'time':time.time()});raise
