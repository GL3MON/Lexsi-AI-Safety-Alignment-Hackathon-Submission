"""Read-only live job logs; checkpoints remain in the mounted private bucket."""
import json,time
from pathlib import Path
from huggingface_hub import HfApi
here=Path(__file__).resolve().parent;job=json.loads((here/'job-long.json').read_text());out=here.parents[1]/'outputs/hf-jobs'/job['job_id'];out.mkdir(parents=True,exist_ok=True);api=HfApi()
for _ in range(240):
 j=api.inspect_job(job_id=job['job_id'],namespace='GL3MON');state=dict(job_id=j.id,status=j.status.stage,message=j.status.message,checked_at=time.time(),checkpoint_bucket=job['checkpoint_bucket'])
 try:
  logs=list(api.fetch_job_logs(job_id=j.id,namespace='GL3MON',follow=False));(out/'job.log').write_text('\n'.join(logs)+'\n')
 except Exception as e:state['log_error']=type(e).__name__
 (out/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
 if j.status.stage in ['COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED']:break
 time.sleep(30)
