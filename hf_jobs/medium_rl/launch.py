import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args();api=HfApi();account=api.whoami()['name'];original=json.loads((ROOT/'hf_jobs/clarification_diagnostic/job.json').read_text());source=json.loads((ROOT/'hf_jobs/clarification_dpo/job.json').read_text());assert account==original['account']==source['account']
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account);h=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert h.unit_label=='minute';reserve=round(181*h.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Account budget exceeded')
 bootstrap=(HERE/'bootstrap.py').read_text();evaluator=json.dumps({f.name:f.read_text() for f in (ROOT/'src/tau2_grpo').glob('*.py')});ast.parse(bootstrap);json.loads(evaluator)
 initial_sha=hashlib.sha256((ROOT/'outputs/rl-initialization/post20-6af9edf523985fb9cd839302/adapter_model.safetensors').read_bytes()).hexdigest();assert initial_sha=='210d9192cfad7e5fb999d70fc2e356a8af88788e10d21865fb7d9a95dc43a983'
 prefix='medium-rl-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 plan=dict(account=account,task='medium-retail-rl',inference_only=False,groups=50,rollouts_per_group=16,mix={'medium_workflows':.6,'opening_clarification':.2,'retention':.2},checkpoint='Post20 checkpoint-7',smoke_groups=3,reward='verified medium progress plus binary short retention',validation_cases=114,validation_every_groups=30,temperature=.7,top_p=1.,response_tokens=192,max_actions=8,baseline='Paired medium and short validation before updates',timeout_minutes=180,reserved_maximum_usd=reserve,remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=original['checkpoint_bucket'],bucket_prefix=prefix,hub_output_uploads=False)
 if not a.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=account).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Medium RL already active')
 archive=HERE/'medium-bank.tar.gz';archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive_prefix='inputs/medium-bank-'+archive_sha[:12]
 api.sync_bucket(str(HERE),'hf://buckets/'+plan['checkpoint_bucket']+'/'+archive_prefix,include=['medium-bank.tar.gz'],quiet=True)
 overlay=base64.b64encode(zlib.compress(evaluator.encode(),9)).decode();overlay_env={f'SOURCE_OVERLAY_{i//48000}':overlay[i:i+48000] for i in range(0,len(overlay),48000)}
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'EVAL_BOOTSTRAP\'])),\'<retail90-eval>\',\'exec\'),{\'__name__\':\'__main__\'})"'];auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=10800,namespace=account,name='medium-retail-grpo-50groups',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False),Volume(type='bucket',source=source['checkpoint_bucket'],mount_path='/source',read_only=True)],env=dict(INITIAL_ADAPTER_SHA256=initial_sha,MEDIUM_ARCHIVE_SHA256=archive_sha,MEDIUM_ARCHIVE_PATH='/checkpoints/'+archive_prefix+'/medium-bank.tar.gz',EVAL_OUTPUT='/checkpoints/'+prefix,SOURCE_INPUT_PREFIX=source['input_prefix'],SOURCE_SHA256=source['source_sha256'],EVAL_CONFIG=(ROOT/'configs/rl-retail-medium.json').read_text(),SOURCE_OVERLAY_PARTS=str(len(overlay_env)),**overlay_env,EVAL_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,flavor='l40sx1');ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
