import argparse,ast,base64,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token
from huggingface_hub._space_api import Volume
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def code(data):return base64.b64encode(zlib.compress(data,9)).decode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args();api=HfApi();p=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(p.read_text());hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';maximum=round(91*hardware.unit_cost_micro_usd/1e6+.01,2);previous=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger)
 if previous+maximum>20:raise RuntimeError('Would exceed original $20 budget')
 overlay={}
 for name in ['rl.py','rl_policy.py','rl_objective.py','rl_transformers_workers.py','rl_short.py','rl_config.py','rl_rollout.py','rl_reporting.py']:
  relative='src/tau2_grpo/'+name;text=(ROOT/relative).read_text();ast.parse(text);overlay[relative]=text
 bootstrap=(HERE/'bootstrap.py').read_text();ast.parse(bootstrap)
 cfg=json.loads((HERE/'config.json').read_text());stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');prefix='short-curriculum-'+stamp
 plan=dict(initial_checkpoint='focused run-2190c37f90/checkpoint-25',groups=16,rollouts_per_group=8,max_actions=4,response_tokens=128,before_after_heldout_greedy_cases=9,environment='short_retail',reward_mode='short_verifiable',native_tau2_scores=False,timeout_minutes=90,reserved_maximum_usd=maximum,prior_committed_usd=round(previous,2),remaining_original_budget_usd=round(20-previous-maximum,2),hub_output_uploads=False,checkpoint_bucket='GL3MON/tau2-retail-rl-checkpoints',bucket_prefix=prefix)
 if not args.submit:print(json.dumps(plan,indent=2));return
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 api.create_bucket(plan['checkpoint_bucket'],private=True,exist_ok=True)
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'RL_BOOTSTRAP_CODE\'])),\'<short-rl-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=5400,namespace='GL3MON',name='retail-short-subgoal-grpo-90min',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False)],env=dict(RL_CHECKPOINT_DIR='/checkpoints/'+prefix,RL_BOOTSTRAP_CODE=code(bootstrap.encode()),RL_SOURCE_OVERLAY=code(json.dumps(overlay).encode()),SHORT_RL_CONFIG=json.dumps(cfg),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,task='short-retail-grpo',flavor='l40sx1');ledger.append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
