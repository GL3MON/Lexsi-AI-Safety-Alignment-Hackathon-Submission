import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args();api=HfApi();account=api.whoami()['name'];original=json.loads((ROOT/'hf_jobs/clarification_diagnostic/job.json').read_text());source=json.loads((ROOT/'hf_jobs/clarification_dpo/job.json').read_text());assert account==original['account']==source['account']
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account);h=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert h.unit_label=='minute';reserve=round(46*h.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Account budget exceeded')
 bootstrap=(HERE/'bootstrap.py').read_text();evaluator=json.dumps({f.name:f.read_text() for f in (ROOT/'src/tau2_grpo').glob('*.py')});ast.parse(bootstrap);json.loads(evaluator)
 parent=json.loads((ROOT/'hf_jobs/midturn_full_rl/job.json').read_text());stop=json.loads((ROOT/'outputs/hf-jobs'/parent['job_id']/'stop-at-20.json').read_text())
 stage=api.inspect_job(job_id=parent['job_id'],namespace=account).status.stage
 if stage not in ('CANCELED','COMPLETED','STOPPED'):raise RuntimeError('Wait for the 20-group stop before evaluating')
 remote=stop['checkpoint_prefix'];local=ROOT/'outputs/rl-initialization'/('post20-'+parent['job_id']);local.mkdir(parents=True,exist_ok=True)
 entries=list(api.list_bucket_tree(parent['checkpoint_bucket'],prefix=remote,recursive=True));assert any(e.path==remote+'/COMPLETE' for e in entries)
 pairs=[(e.path,local/e.path.removeprefix(remote+'/policy/')) for e in entries if e.path.startswith(remote+'/policy/') and e.path.endswith(('.json','.jinja','.safetensors'))];api.download_bucket_files(parent['checkpoint_bucket'],pairs)
 initial_sha=hashlib.sha256((local/'adapter_model.safetensors').read_bytes()).hexdigest()
 prefix='post20-eval-resumed-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 plan=dict(account=account,task='post20-paired-eval',inference_only=True,groups=0,rollouts_per_group=1,mix={'midturn':.6,'opening_clarification':.2,'retention':.2},checkpoint='Retained checkpoint after 20 completed rollout groups',resume_from='post20-eval-20261115T181717/evaluation',reuse_complete_episodes=True,validation_cases=114,tau2_cases=3,tau2_max_actor_actions=20,temperature=0.,top_p=1.,response_tokens=192,max_actions=4,baseline='Paired pilot checkpoint-6 on identical held-out cases',timeout_minutes=45,reserved_maximum_usd=reserve,remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=original['checkpoint_bucket'],bucket_prefix=prefix,hub_output_uploads=False)
 if not a.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=account).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Paired evaluation already active')
 overlay=base64.b64encode(zlib.compress(evaluator.encode(),9)).decode();overlay_env={f'SOURCE_OVERLAY_{i//48000}':overlay[i:i+48000] for i in range(0,len(overlay),48000)}
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'EVAL_BOOTSTRAP\'])),\'<retail90-eval>\',\'exec\'),{\'__name__\':\'__main__\'})"'];auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=2700,namespace=account,name='post20-resume-tau2-validation',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False),Volume(type='bucket',source=source['checkpoint_bucket'],mount_path='/source',read_only=True)],env=dict(INITIAL_ADAPTER_SHA256=initial_sha,EVAL_INITIAL_PATH='/checkpoints/'+remote+'/policy',TAU2_SPLIT_MANIFEST=(ROOT/'outputs/retail-data/split_manifest.json').read_text(),EVAL_OUTPUT='/checkpoints/'+prefix,SOURCE_INPUT_PREFIX=source['input_prefix'],SOURCE_SHA256=source['source_sha256'],EVAL_CONFIG=(ROOT/'configs/rl-retail-midturn-full.json').read_text(),SOURCE_OVERLAY_PARTS=str(len(overlay_env)),**overlay_env,EVAL_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,initial_adapter_sha256=initial_sha,initial_checkpoint_prefix=remote,tau2_tasks=['69','34','63'],job_id=job.id,url=job.url,flavor='l40sx1');ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
