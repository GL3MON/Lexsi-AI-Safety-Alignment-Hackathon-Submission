"""Cancel the existing job only after the requested number of completed groups."""
import argparse,json,time,fcntl
from pathlib import Path
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--groups',type=int,default=20);a=p.parse_args();lock=(HERE/'stop-after.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 j=json.loads((HERE/'job.json').read_text());api=HfApi();out=ROOT/'outputs/hf-jobs'/j['job_id'];out.mkdir(parents=True,exist_ok=True)
 remote=j['bucket_prefix']+'/training/training/groups.json';local=out/'stop-watcher-groups.json'
 while True:
  try:
   state=api.inspect_job(job_id=j['job_id'],namespace=j['account']).status.stage
   if state in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):
    print(json.dumps(dict(event='job_already_finished',status=state)),flush=True);return
   entries=list(api.list_bucket_tree(j['checkpoint_bucket'],prefix=j['bucket_prefix']+'/training/training',recursive=False))
   if any(e.path==remote for e in entries):
    api.download_bucket_files(j['checkpoint_bucket'],[(remote,local)])
    rows=json.loads(local.read_text());print(json.dumps(dict(event='group_limit_watch',completed_groups=len(rows),limit=a.groups)),flush=True)
    if len(rows)>=a.groups:
     # groups.json is persisted after checkpoint saving: completed updates are durable.
     versions=[r.get('policy_version',0)+r.get('optimizer_steps',0) for r in rows]
     checkpoint=max(versions)
     note=dict(job_id=j['job_id'],requested_group_limit=a.groups,completed_groups=len(rows),latest_checkpoint=checkpoint,checkpoint_prefix=j['bucket_prefix']+f'/training/training/checkpoint-{checkpoint}',reason='User requested stop after 20 rollout groups',time=time.time())
     (out/'stop-at-20.json').write_text(json.dumps(note,indent=2)+'\n')
     api.cancel_job(job_id=j['job_id'],namespace=j['account']);print(json.dumps(dict(event='job_canceled_at_group_limit',**note)),flush=True);return
  except Exception as e:print(json.dumps(dict(event='watch_error',error=str(e))),flush=True)
  time.sleep(10)
if __name__=='__main__':main()
