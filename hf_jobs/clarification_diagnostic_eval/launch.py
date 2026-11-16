"""Corrected evaluation only; reuse saved weights, no retraining or input upload."""
import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args();api=HfApi();account=api.whoami()['name'];original=json.loads((ROOT/'hf_jobs/clarification_diagnostic/job.json').read_text());assert account==original['account']
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 p=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(p.read_text());previous=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger if r.get('account')==account);h=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert h.unit_label=='minute';reserve=round(16*h.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Account budget exceeded')
 bootstrap=(HERE/'bootstrap.py').read_text();fixed=(ROOT/'src/tau2_grpo/sft_diagnostic.py').read_text();ast.parse(bootstrap);ast.parse(fixed)
 initial=ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19';files={n:hashlib.sha256((initial/n).read_bytes()).hexdigest() for n in ['adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']}
 prefix='sft-diagnostic-corrected-eval-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 plan=dict(account=account,task='sft-decision-diagnostic-eval',inference_only=True,training_job=original['job_id'],before_checkpoint='SFT19',after_checkpoint='Diagnostic SFT50',evaluations=160,timeout_minutes=15,reserved_maximum_usd=reserve,prior_account_committed_usd=round(previous,2),remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=original['checkpoint_bucket'],bucket_prefix=prefix,fix='Fresh controller per independent case; exact case-message and unique-prompt checks',hub_output_uploads=False)
 if not args.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());state=api.inspect_job(job_id=old['job_id'],namespace=account).status.stage
  if state not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Corrected evaluation already active')
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'DIAG_BOOTSTRAP\'])),\'<fixed-diagnostic-eval>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=900,namespace=account,name='retail-sft-diagnostic-corrected-eval-only',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False)],env=dict(DIAG_OUTPUT_DIR='/checkpoints/'+prefix,DIAG_INPUT_PREFIX=original['input_prefix'],DIAG_INPUT_SHA256=original['source_sha256'],DIAG_INITIAL_FILES=json.dumps(files),DIAG_EVAL_CONFIG=(ROOT/'hf_jobs/short_bank_rl/config.json').read_text(),FIXED_EVALUATOR=base64.b64encode(zlib.compress(fixed.encode(),9)).decode(),DIAG_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token()))
 entry=dict(plan,job_id=job.id,url=job.url,flavor='l40sx1');ledger.append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
