"""Read-only monitoring for the SFT diagnostic in its own account namespace."""
import json,time
from pathlib import Path
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];job=json.loads((HERE/'job.json').read_text());api=HfApi();out=ROOT/'outputs/hf-jobs'/job['job_id'];out.mkdir(parents=True,exist_ok=True)
for _ in range(240):
 current=api.inspect_job(job_id=job['job_id'],namespace=job['account']);state=dict(job_id=current.id,status=current.status.stage,message=current.status.message,checked_at=time.time(),checkpoint_bucket=job['checkpoint_bucket'],bucket_prefix=job['bucket_prefix'],account=job['account'])
 try:(out/'job.log').write_text('\n'.join(api.fetch_job_logs(job_id=current.id,namespace=job['account'],follow=False))+'\n')
 except Exception as error:state['log_error']=type(error).__name__
 try:
  pairs=[]
  for e in api.list_bucket_tree(job['checkpoint_bucket'],prefix=job['bucket_prefix'],recursive=True):
   name=e.path;leaf=name.rsplit('/',1)[-1]
   if '..' in Path(name).parts:continue
   if leaf in ['summary.json','result.json','selection.json','progress.json','baseline-preference-eval.json','runtime.json','error.txt'] or name.endswith('.txt'):
    dest=out/'bucket'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if name.endswith('.txt') and dest.exists():continue
    pairs.append((name,dest))
  if pairs:api.download_bucket_files(job['checkpoint_bucket'],pairs,raise_on_missing_files=False)
 except Exception as error:state['sync_error']=type(error).__name__
 (out/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
 if current.status.stage in ['COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED']:break
 time.sleep(30)
