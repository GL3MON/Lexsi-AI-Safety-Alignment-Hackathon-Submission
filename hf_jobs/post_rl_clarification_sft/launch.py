"""Budget-bounded SFT from final short-RL28, with retention and post-SFT validation."""
import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token
from huggingface_hub._space_api import Volume
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def code(data):return base64.b64encode(zlib.compress(data,9)).decode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args();api=HfApi()
 data=ROOT/'outputs/retail-data/post-rl-clarification200-retention100';manifest=json.loads((data/'manifest.json').read_text());assert manifest['complete'] and manifest['training_records']==300 and manifest['verified_records']==344
 for split,sha in manifest['data_sha256'].items():assert hashlib.sha256((data/(split+'.jsonl')).read_bytes()).hexdigest()==sha
 assert hashlib.sha256((ROOT/'outputs/rl-initialization/short-bank-checkpoint-28/adapter_model.safetensors').read_bytes()).hexdigest()==manifest['initial_adapter_sha256']
 ledger_file=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_file.read_text());hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';maximum=round(26*hardware.unit_cost_micro_usd/1e6+.01,2);previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger)
 if previous+maximum>20:raise RuntimeError('Would exceed original $20 budget')
 bootstrap=(HERE/'bootstrap.py').read_text();ast.parse(bootstrap)
 for p in (ROOT/'src/tau2_grpo').glob('*.py'):ast.parse(p.read_text())
 sft=json.loads((ROOT/'configs/sft-post-rl-clarification200-retention100.json').read_text());rl=json.loads((ROOT/'hf_jobs/short_bank_rl/config.json').read_text())
 archive=HERE/'curated-input.tar.gz';sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive_path='inputs/post-rl-clarification-'+sha[:16]+'.tar.gz';bank=ROOT/'hf_jobs/short_bank_rl/scenario-bank.tar.gz';bank_sha=hashlib.sha256(bank.read_bytes()).hexdigest()
 prefix='post-rl-clarification-sft-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 plan=dict(task='post-rl28-clarification-sft-with-retention',initial_checkpoint=28,initial_adapter_sha256=manifest['initial_adapter_sha256'],training_records=300,clarification_records=200,retention_records=100,validation_records=44,batch_size=sft['batch_size'],gradient_accumulation_steps=sft['gradient_accumulation_steps'],epochs=1,planned_optimizer_steps=__import__('math').ceil(300/(sft['batch_size']*sft['gradient_accumulation_steps'])),learning_rate=1e-5,eval_steps=sft['eval_steps'],save_total_limit=2,post_sft_greedy_cases=30,timeout_minutes=25,reserved_maximum_usd=maximum,prior_committed_usd=round(previous,2),remaining_original_budget_usd=round(20-previous-maximum,2),hub_output_uploads=False,checkpoint_bucket='GL3MON/tau2-retail-rl-checkpoints',bucket_prefix=prefix,input_sha256=sha)
 if not args.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());state=api.inspect_job(job_id=old['job_id'],namespace='GL3MON').status.stage
  if state not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('An SFT job is already active')
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 if not any(e.path==archive_path for e in api.list_bucket_tree(plan['checkpoint_bucket'],prefix='inputs',recursive=False)):
  api.batch_bucket_files(plan['checkpoint_bucket'],add=[(archive,archive_path)])
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'RL_BOOTSTRAP_CODE\'])),\'<post-rl-sft-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=1500,namespace='GL3MON',name='retail-post-rl28-clarification-retention-sft',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False)],env=dict(CURATED_ARCHIVE_PATH=archive_path,CURATED_ARCHIVE_SHA256=sha,BANK_ARCHIVE_PATH='inputs/short-bank-'+bank_sha[:16]+'.tar.gz',BANK_ARCHIVE_SHA256=bank_sha,INITIAL_ADAPTER_SHA256=manifest['initial_adapter_sha256'],RL_CHECKPOINT_DIR='/checkpoints/'+prefix,RL_BOOTSTRAP_CODE=code(bootstrap.encode()),RL_SOURCE_OVERLAY=code(b'{}'),POST_RL_SFT_CONFIG=json.dumps(sft),SHORT_RL_CONFIG=json.dumps(rl),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,flavor='l40sx1');ledger.append(entry);ledger_file.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
