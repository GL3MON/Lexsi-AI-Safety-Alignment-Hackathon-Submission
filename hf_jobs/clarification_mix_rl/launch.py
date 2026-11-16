"""Continued GRPO from selected SFT19; explicit budget cap and private bucket only."""
import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token
from huggingface_hub._space_api import Volume
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def code(data):return base64.b64encode(zlib.compress(data,9)).decode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');parser.add_argument('--budget-cap',type=float,default=20);parser.add_argument('--timeout-minutes',type=int,default=90);args=parser.parse_args()
 if not 1<=args.timeout_minutes<=240:parser.error('Timeout must be between 1 and 240 minutes')
 api=HfApi();p=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(p.read_text());hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';maximum=round((args.timeout_minutes+1)*hardware.unit_cost_micro_usd/1e6+.01,2);previous=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger)
 overlay={}
 for name in ['rl.py','rl_policy.py','rl_objective.py','rl_transformers_workers.py','rl_short.py','rl_config.py','rl_rollout.py','rl_reporting.py','rl_bank.py','short_scenario_bank.py']:
  relative='src/tau2_grpo/'+name;text=(ROOT/relative).read_text();ast.parse(text);overlay[relative]=text
 bootstrap=(HERE/'bootstrap.py').read_text();ast.parse(bootstrap)
 cfg=json.loads((ROOT/'configs/rl-retail-clarification-mix.json').read_text());assert cfg['group_size']==cfg['rollout_workers']==cfg['rollout_batch_size']==16
 sha=hashlib.sha256((ROOT/cfg['initial_adapter']/'adapter_model.safetensors').read_bytes()).hexdigest();assert sha=='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811'
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');prefix='clarification-mix-rl16-'+stamp
 archive=ROOT/'hf_jobs/short_bank_rl/scenario-bank.tar.gz';archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest();archive_path='inputs/short-bank-'+archive_hash[:16]+'.tar.gz'
 plan=dict(initial_checkpoint='selected clarification-retention SFT checkpoint19, continued from RL28',initial_adapter_sha256=sha,groups=cfg['train_steps'],rollouts_per_group=16,target_training_episodes=cfg['train_steps']*16,clarification_groups=60,retention_groups=30,max_actions=4,response_tokens=192,greedy_validation_cases=30,environment='short_bank',reward_mode='short_verifiable',native_tau2_scores=False,timeout_minutes=args.timeout_minutes,reserved_maximum_usd=maximum,prior_committed_usd=round(previous,2),proposed_cumulative_maximum_usd=round(previous+maximum,2),authorized_budget_cap_usd=args.budget_cap,within_budget=previous+maximum<=args.budget_cap,remaining_original_budget_usd=round(20-previous,2),remaining_after_reservation_usd=round(args.budget_cap-previous-maximum,2),hub_output_uploads=False,checkpoint_bucket='GL3MON/tau2-retail-rl-checkpoints',bucket_prefix=prefix,scenario_bank_sha256=archive_hash,validation_every_groups=30)
 if not args.submit:
  (HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2));return
 if not plan['within_budget']:raise RuntimeError('Run exceeds authorized budget. Obtain explicit approval for a larger cumulative cap before submitting.')
 if (HERE/'job.json').exists():
  previous_job=json.loads((HERE/'job.json').read_text());state=api.inspect_job(job_id=previous_job['job_id'],namespace='GL3MON').status.stage
  if state not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Continued RL job already active')
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'RL_BOOTSTRAP_CODE\'])),\'<clarification-rl-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=args.timeout_minutes*60,namespace='GL3MON',name='retail-clarification-mix-grpo-16rollouts',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False)],env=dict(BANK_ARCHIVE_PATH=archive_path,BANK_ARCHIVE_SHA256=archive_hash,RL_CHECKPOINT_DIR='/checkpoints/'+prefix,RL_BOOTSTRAP_CODE=code(bootstrap.encode()),RL_SOURCE_OVERLAY=code(json.dumps(overlay).encode()),SHORT_RL_CONFIG=json.dumps(cfg),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,task='continued-clarification-mix-grpo16',flavor='l40sx1');ledger.append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
