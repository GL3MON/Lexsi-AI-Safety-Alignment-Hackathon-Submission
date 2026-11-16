"""Submit one bounded evaluation after the existing 20-group watcher stops training."""
import fcntl,json,time,subprocess
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
lock=(HERE/'launch-after-stop.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);parent=json.loads((ROOT/'hf_jobs/midturn_full_rl/job.json').read_text());note=ROOT/'outputs/hf-jobs'/parent['job_id']/'stop-at-20.json';api=HfApi()
while True:
 state=api.inspect_job(job_id=parent['job_id'],namespace=parent['account']).status.stage
 if state in ('COMPLETED','CANCELED','STOPPED','ERROR','TIMEOUT'):
  if not note.exists():raise RuntimeError('Training stopped without the requested 20-group checkpoint marker')
  if (HERE/'job.json').exists():raise RuntimeError('An evaluation has already been submitted; refusing a duplicate')
  print('Training stopped; submitting paired evaluation',flush=True)
  subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'launch.py'),'--submit'],cwd=ROOT,check=True)
  subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'watch.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=open(ROOT/'outputs/logs/post20-eval-watch.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
  break
 print(json.dumps(dict(event='wait_for_group20_stop',status=state)),flush=True);time.sleep(10)
