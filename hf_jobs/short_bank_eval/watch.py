"""Read-only job/rollout monitoring; no Hub commits or output uploads."""
import json,time
from pathlib import Path
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];job=json.loads((HERE/'job.json').read_text());out=ROOT/'outputs/hf-jobs'/job['job_id'];out.mkdir(parents=True,exist_ok=True);api=HfApi()
for _ in range(240):
 current=api.inspect_job(job_id=job['job_id'],namespace='GL3MON');state=dict(job_id=current.id,status=current.status.stage,message=current.status.message,checked_at=time.time(),checkpoint_bucket=job['checkpoint_bucket'],bucket_prefix=job['bucket_prefix'])
 try:
  lines=list(api.fetch_job_logs(job_id=current.id,namespace='GL3MON',follow=False));(out/'job.log').write_text('\n'.join(lines)+'\n')
 except Exception as error:state['log_error']=type(error).__name__
 try:
  downloads=[]
  for entry in api.list_bucket_tree(job['checkpoint_bucket'],prefix=job['bucket_prefix'],recursive=True):
   name=getattr(entry,'path','');leaf=name.rsplit('/',1)[-1]
   if '..' in Path(name).parts:continue
   if leaf in ['summary.json','groups.json','result.json','runtime.json','error.txt'] or name.endswith('.txt'):
    dest=out/'bucket'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if name.endswith('.txt') and dest.exists():continue
    downloads.append((name,dest))
  if downloads:api.download_bucket_files(job['checkpoint_bucket'],downloads,raise_on_missing_files=False)
 except Exception:pass
 (out/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
 if current.status.stage in ['COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED']:break
 time.sleep(30)
