"""Launch private portable DPO under the currently funded account; no Hub commits."""
import argparse,ast,base64,hashlib,json,netrc,tarfile,zlib,math
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args();api=HfApi();account=api.whoami()['name']
 if account=='GL3MON':raise RuntimeError('Funded alternate account must be signed in')
 # Visibility alone is insufficient: verify the signed-in account can read gated files.
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 ledger_file=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_file.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account);hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';reserve=round(91*hardware.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Would exceed the new account $20 budget')
 cfg=json.loads((ROOT/'configs/dpo-clarification.json').read_text());cfg['epochs']=1
 from tau2_grpo.dpo import load_settings,read_pairs
 load_settings(ROOT/'configs/dpo-clarification.json')
 data=ROOT/'outputs/retail-data/clarification-dpo';manifest=json.loads((data/'manifest.json').read_text());assert manifest['complete']
 for split,sha in manifest['data_sha256'].items():assert hashlib.sha256((data/(split+'.jsonl')).read_bytes()).hexdigest()==sha
 initial=ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19';files={n:hashlib.sha256((initial/n).read_bytes()).hexdigest() for n in ['adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']};assert files['adapter_model.safetensors']==manifest['initial_adapter_sha256']
 bootstrap=(HERE/'bootstrap.py').read_text();ast.parse(bootstrap)
 for p in (ROOT/'src/tau2_grpo').glob('*.py'):ast.parse(p.read_text())
 archive=HERE/'source.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  with tarfile.open(ROOT/'outputs/account-transfer/payload.tar.gz') as old:
   for member in old:
    if member.name.startswith('tau2-bench/'):
     tar.addfile(member,old.extractfile(member) if member.isfile() else None)
  tar.add(ROOT/'pyproject.toml',arcname='project/pyproject.toml')
  for p in (ROOT/'src/tau2_grpo').glob('*.py'):tar.add(p,arcname='project/src/tau2_grpo/'+p.name)
  tar.add(ROOT/'templates',arcname='project/templates')
  for name in ['train.jsonl','validation.jsonl','manifest.json']:tar.add(data/name,arcname='project/preferences/'+name)
  tar.add(ROOT/'hf_jobs/short_bank_rl/scenario-bank.tar.gz',arcname='project/scenario-bank.tar.gz')
  tar.add(ROOT/'outputs/hf-jobs/6af9a95229dd05ff0368b092/bucket/weighted-clarification-sft-20261115T124500/short-before-weighted/summary.json',arcname='project/baseline-summary.json')
 sha=hashlib.sha256(archive.read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');prefix='dpo-clarification-'+stamp;input_prefix='inputs/'+prefix;bucket=account+'/tau2-dpo-artifacts'
 plan=dict(account=account,initial_checkpoint='SFT19',initial_adapter_sha256=files['adapter_model.safetensors'],train_pairs=300,eval_pairs=68,categories={'missing_ask':100,'supplied_call':100,'retention':100},epochs=1,pairs_per_batch=cfg['pairs_per_batch'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],planned_optimizer_steps=math.ceil(math.ceil(300/cfg['pairs_per_batch'])/cfg['gradient_accumulation_steps']),beta=cfg['beta'],learning_rate=cfg['learning_rate'],before_reference_greedy_score='57/90 saved SFT19 baseline',after_greedy_cases=90,timeout_minutes=90,reserved_maximum_usd=reserve,prior_account_committed_usd=round(previous,2),remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=bucket,bucket_prefix=prefix,input_prefix=input_prefix,source_sha256=sha,hub_output_uploads=False,authorization='User stated other account has $20, signed it in, and requested DPO launch')
 if not args.submit:(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=old['account']).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('DPO job already active')
 api.create_bucket(bucket,private=True,exist_ok=True)
 print('TRANSFERRING_PRIVATE_INPUTS',account,flush=True)
 api.batch_bucket_files(bucket,add=[(archive,input_prefix+'/source.tar.gz')]+[(initial/n,input_prefix+'/initial-adapter/'+n) for n in files])
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'DPO_BOOTSTRAP\'])),\'<dpo-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 eval_cfg=json.loads((ROOT/'hf_jobs/short_bank_rl/config.json').read_text())
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=5400,namespace=account,name='retail-clarification-dpo-from-sft19',volumes=[Volume(type='bucket',source=bucket,mount_path='/checkpoints',read_only=False)],env=dict(DPO_OUTPUT_DIR='/checkpoints/'+prefix,DPO_INPUT_PREFIX=input_prefix,DPO_INPUT_SHA256=sha,DPO_INITIAL_FILES=json.dumps(files),DPO_CONFIG=json.dumps(cfg),DPO_EVAL_CONFIG=json.dumps(eval_cfg),DPO_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,task='clarification-dpo',flavor='l40sx1');ledger.append(entry);ledger_file.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
