"""Small matched-pair SFT memorization/transfer diagnostic on the funded account."""
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
 ledger_file=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_file.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account);hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';reserve=round(31*hardware.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Would exceed the new account $20 budget')
 cfg=json.loads((ROOT/'configs/sft-clarification-diagnostic20.json').read_text())
 from tau2_grpo.sft_config import load_settings
 load_settings(ROOT/'configs/sft-clarification-diagnostic20.json')
 data=ROOT/'outputs/retail-data/clarification-diagnostic20';manifest=json.loads((data/'manifest.json').read_text());assert manifest['complete']
 for split,sha in manifest['data_sha256'].items():assert hashlib.sha256((data/(split+'.jsonl')).read_bytes()).hexdigest()==sha
 initial=ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19';files={n:hashlib.sha256((initial/n).read_bytes()).hexdigest() for n in ['adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']};assert files['adapter_model.safetensors']=='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811'
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
  for name in ['train.jsonl','heldout.jsonl','manifest.json']:tar.add(data/name,arcname='project/diagnostic/'+name)
  tar.add(data/'worlds',arcname='project/diagnostic/worlds')
 sha=hashlib.sha256(archive.read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');prefix='sft-decision-diagnostic-'+stamp;input_prefix='inputs/'+prefix;bucket=account+'/tau2-retail-experiments'
 plan=dict(account=account,initial_checkpoint='SFT19',initial_adapter_sha256=files['adapter_model.safetensors'],training_pairs=20,training_examples=40,heldout_pairs=20,heldout_examples=40,fields=['email','first_name','last_name','zip'],max_steps=cfg['max_steps'],batch_size=cfg['batch_size'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],learning_rate=cfg['learning_rate'],objective='Equal per-example mean NLL, ask/call weights1:1',purpose='Memorization and unseen-identity diagnostic only',timeout_minutes=30,reserved_maximum_usd=reserve,prior_account_committed_usd=round(previous,2),remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=bucket,bucket_prefix=prefix,input_prefix=input_prefix,source_sha256=sha,hub_output_uploads=False,authorization='User explicitly authorized HF for the small SFT diagnostic')
 if not args.submit:(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=old['account']).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Diagnostic job already active')
 api.create_bucket(bucket,private=True,exist_ok=True)
 print('TRANSFERRING_PRIVATE_INPUTS',account,flush=True)
 api.batch_bucket_files(bucket,add=[(archive,input_prefix+'/source.tar.gz')]+[(initial/n,input_prefix+'/initial-adapter/'+n) for n in files])
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'DIAG_BOOTSTRAP\'])),\'<sft-diagnostic-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 eval_cfg=json.loads((ROOT/'hf_jobs/short_bank_rl/config.json').read_text())
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=1800,namespace=account,name='retail-sft-ask-call-small-diagnostic',volumes=[Volume(type='bucket',source=bucket,mount_path='/checkpoints',read_only=False)],env=dict(DIAG_OUTPUT_DIR='/checkpoints/'+prefix,DIAG_INPUT_PREFIX=input_prefix,DIAG_INPUT_SHA256=sha,DIAG_INITIAL_FILES=json.dumps(files),DIAG_SFT_CONFIG=json.dumps(cfg),DIAG_EVAL_CONFIG=json.dumps(eval_cfg),DIAG_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,task='sft-decision-diagnostic',flavor='l40sx1');ledger.append(entry);ledger_file.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
