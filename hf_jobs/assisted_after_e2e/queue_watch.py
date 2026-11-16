"""One-shot SFT -> completed rollout -> assisted RL dependency watcher."""
import fcntl,json,math,subprocess,time,traceback
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;KEY='retail-e2e-followup'
TERMINAL={'COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'}
def readiness(stage,finished,error,rollout):
 if stage not in TERMINAL:return False,'waiting_for_sft_and_rollout'
 if stage!='COMPLETED' or error or not finished:return False,'blocked_sft_or_rollout_job_failed'
 if not rollout or not rollout.get('complete'):return False,'blocked_rollout_unscored'
 return True,'ready'
def record(**data):
 (HERE/'queue-status.json').write_text(json.dumps(dict(updated_at=time.time(),**data),indent=2)+'\n');print(json.dumps(data),flush=True)
def settle_parent(api,j):
 info=api.inspect_job(job_id=j['job_id'],namespace=j['account']);elapsed=(info.finished_at.timestamp()-info.started_at.timestamp()) if info.started_at and info.finished_at else 90*60;bound=round((math.ceil(elapsed/60)+2)*.03+.01,2);p=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';rows=json.loads(p.read_text())
 for r in rows:
  if r.get('job_id')==j['job_id']:r.update(status=info.status.stage,settled_maximum_usd=min(bound,r['reserved_maximum_usd']))
 p.write_text(json.dumps(rows,indent=2)+'\n')
def release():
 p=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';rows=json.loads(p.read_text())
 for r in rows:
  if r.get('reservation_key')==KEY:r.update(status='BLOCKED',settled_maximum_usd=0.)
 p.write_text(json.dumps(rows,indent=2)+'\n')
def main():
 lock=(HERE/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);api=HfApi();j=json.loads((ROOT/'hf_jobs/retail_e2e_sft/job.json').read_text());local=HERE/'parent-artifacts';local.mkdir(exist_ok=True)
 while True:
  try:
   if (HERE/'job.json').exists():record(status='ALREADY_SUBMITTED',job=json.loads((HERE/'job.json').read_text()));return
   info=api.inspect_job(job_id=j['job_id'],namespace=j['account']);stage=info.status.stage
   if stage not in TERMINAL:record(status='WAITING',parent_job=j['job_id'],parent_stage=stage);time.sleep(20);continue
   settle_parent(api,j);paths={e.path for e in api.list_bucket_tree(j['checkpoint_bucket'],prefix=j['bucket_prefix'],recursive=True)};prefix=j['bucket_prefix'];finished=prefix+'/JOB_FINISHED' in paths;error=prefix+'/error.txt' in paths;rollout=None
   if prefix+'/tau2-test/summary.json' in paths:
    api.download_bucket_files(j['checkpoint_bucket'],[(prefix+'/tau2-test/summary.json',local/'rollout-summary.json')]);rollout=json.loads((local/'rollout-summary.json').read_text())
   ready,reason=readiness(stage,finished,error,rollout)
   if not ready:release();record(status='BLOCKED',reason=reason,parent_job=j['job_id']);return
   record(status='SUBMITTING',parent_job=j['job_id'],rollout_native_reward=rollout.get('native_reward'))
   subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'launch.py'),'--submit'],cwd=ROOT,check=True)
   child=json.loads((HERE/'job.json').read_text());record(status='SUBMITTED',job_id=child['job_id'],url=child['url'],initial_adapter_sha256=child['initial_adapter_sha256'])
   with open(ROOT/'outputs/logs/assisted-after-e2e-watch.log','a') as log:subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'watch.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   # Refresh only our viewer; preserve previous sources without touching GPU jobs.
   import os,signal
   saved=ROOT/'outputs/logs/e2e-viewer.pid'
   if saved.exists():
    pid=int(saved.read_text())
    try:
     cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
     if b'scripts/serve_rollouts.py' in cmd:os.kill(pid,signal.SIGTERM)
    except (OSError,ValueError):pass
   args=[str(ROOT/'.venv/bin/python'),'-u','scripts/serve_rollouts.py','--job',str(HERE/'job.json'),'--job','hf_jobs/retail_e2e_sft/job.json','--job','hf_jobs/tau2_assisted_rl/job.json','--job','hf_jobs/midturn_full_rl/job.json']
   with open(ROOT/'outputs/logs/rollout-viewer.log','a') as log:p=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);saved.write_text(str(p.pid))
   return
  except Exception as e:
   record(status='ERROR',error=str(e),traceback=traceback.format_exc());return
if __name__=='__main__':main()
