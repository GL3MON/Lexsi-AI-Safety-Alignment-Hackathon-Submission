"""Launch a budget-bounded real Tau2 GRPO trial from the evaluated focused checkpoint25."""
import argparse,json,netrc,math
from pathlib import Path
from huggingface_hub import HfApi,get_token
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true');parser.add_argument('--timeout-minutes',type=int,default=17);args=parser.parse_args();assert 1<=args.timeout_minutes<=25;api=HfApi()
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text())
 h=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert h.unit_label=='minute';rate=h.unit_cost_micro_usd/1e6
 for row in ledger:
  if 'settled_maximum_usd' in row:continue
  job=api.inspect_job(job_id=row['job_id'],namespace='GL3MON')
  if job.status.stage in ['COMPLETED','ERROR','CANCELED'] and job.durations and job.durations.total_secs:
   seconds=job.durations.total_secs;row.update(settled_maximum_usd=round((math.ceil(seconds/60)+1)*rate+.01,2),provider_total_seconds=seconds,settlement_basis='Terminal provider total duration rounded up, plus one minute and $0.01 provision; not an invoice')
 ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
 previous=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger);maximum=round((args.timeout_minutes+1)*rate+.01,2)
 if previous+maximum>15:raise RuntimeError(f'Protected $5 reserve: committed={previous:.2f}, requested={maximum:.2f}')
 ready=json.loads((HERE/'artifacts/environment-preflight.json').read_text());assert ready['ready'] and ready['checkpoint']==25 and ready['rl_partition_tasks']==22
 input_repo='GL3MON/tau2-retail-focused25-rl-input';result_repo='GL3MON/tau2-retail-focused25-rl-results'
 for repo in [input_repo,result_repo]:assert api.repo_info(repo,repo_type='dataset').private
 plan=dict(initial_checkpoint='focused run-2190c37f90/checkpoint-25',groups=2,rollouts_per_group=8,max_actions=10,rollout_engine='transformers',reward_mode='additive_grounded',maximum_reserved_usd=maximum,prior_committed_upper_usd=round(previous,2),protected_reserve_usd=5,timeout_minutes=args.timeout_minutes,results_repo=result_repo)
 if not args.submit:print(json.dumps(plan,indent=2));return
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import os,runpy;from huggingface_hub import hf_hub_download;runpy.run_path(hf_hub_download(os.environ[\'RL_INPUT_REPO\'],\'bootstrap.py\',repo_type=\'dataset\'),run_name=\'__main__\')"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=args.timeout_minutes*60,namespace='GL3MON',name='tau2-retail-focused25-grpo-trial',env=dict(RL_INPUT_REPO=input_repo,RL_RESULT_REPO=result_repo,RL_JOB_MODE='train',RL_UPLOAD_OUTPUTS='0',WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,reserved_maximum_usd=maximum,task='tau2-retail-focused25-grpo-trial',flavor='l40sx1')
 ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
