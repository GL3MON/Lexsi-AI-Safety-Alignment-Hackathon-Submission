"""Launch verified end-to-end SFT and exactly one native Tau2 rollout."""
import argparse,ast,base64,hashlib,json,netrc,tarfile,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args();api=HfApi();account=api.whoami()['name'];source=json.loads((ROOT/'hf_jobs/clarification_dpo/job.json').read_text());assert account==source['account']=='notGL3MON'
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 from tau2_grpo.sft_config import load_settings
 cfg=load_settings(ROOT/'configs/sft-retail-e2e.json');data=ROOT/'outputs/retail-data/retail-e2e-sft-retention';manifest=json.loads((data/'manifest.json').read_text());audit=json.loads((ROOT/'outputs/diagnostics/retail-e2e-sft-mask-audit/independent-label-audit.json').read_text());assert audit['records_checked']==audit['passed']==703 and not audit['failures'];assert manifest['train']==599 and manifest['validation']==104
 for name,sha in manifest['data_sha256'].items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account);hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');reserve=round(91*hardware.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Insufficient account budget including queued reservations')
 bootstrap=(HERE/'bootstrap.py').read_text();ast.parse(bootstrap);overlay=json.dumps({f.name:f.read_text() for f in (ROOT/'src/tau2_grpo').glob('*.py')});initial_sha=hashlib.sha256((ROOT/'outputs/rl-initialization/post20-6af9edf523985fb9cd839302/adapter_model.safetensors').read_bytes()).hexdigest();assert initial_sha=='210d9192cfad7e5fb999d70fc2e356a8af88788e10d21865fb7d9a95dc43a983'
 prefix='retail-e2e-sft-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');bucket=account+'/tau2-retail-experiments';plan=dict(account=account,task='retail-e2e-sft',initial_checkpoint='Post20 checkpoint-7',initial_adapter_sha256=initial_sha,train_examples=599,validation_examples=104,epochs=2,batch_size=8,gradient_accumulation_steps=2,estimated_optimizer_steps=76,learning_rate=2e-5,eval_steps=10,save_total_limit=2,max_context=8192,post_training_rollout=dict(task_id='69',unassisted=True,temperature=0.),timeout_minutes=90,reserved_maximum_usd=reserve,remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=bucket,bucket_prefix=prefix,hub_output_uploads=False,then_exit=True,followup_queued='assisted RL watcher after checkpoint and rollout validation')
 if not a.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=account).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('End-to-end SFT is already active')
 archive=HERE/'data.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for name in ['train.jsonl','validation.jsonl','task-split-manifest.json','manifest.json','longest-tokenized.json']:t.add(data/name,arcname=name)
  t.add(HERE/'probe.py',arcname='probe.py')
 sha=hashlib.sha256(archive.read_bytes()).hexdigest();remote='inputs/e2e-sft-'+sha[:12]+'/data.tar.gz';api.batch_bucket_files(bucket,add=[(archive,remote)])
 encoded=base64.b64encode(zlib.compress(overlay.encode(),9)).decode();parts={f'SOURCE_OVERLAY_{i//48000}':encoded[i:i+48000] for i in range(0,len(encoded),48000)};auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
 cmd=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'EVAL_BOOTSTRAP\'])),\'<e2e-sft>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=cmd,flavor='l40sx1',timeout=5400,namespace=account,name='retail-end-to-end-sft-then-rollout',volumes=[Volume(type='bucket',source=bucket,mount_path='/checkpoints',read_only=False),Volume(type='bucket',source=source['checkpoint_bucket'],mount_path='/source',read_only=True)],env=dict(EVAL_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),EVAL_OUTPUT='/checkpoints/'+prefix,SOURCE_INPUT_PREFIX=source['input_prefix'],SOURCE_SHA256=source['source_sha256'],SOURCE_OVERLAY_PARTS=str(len(parts)),**parts,SFT_DATA_ARCHIVE='/checkpoints/'+remote,SFT_DATA_SHA256=sha,INITIAL_ADAPTER_SHA256=initial_sha,SFT_CONFIG=(ROOT/'configs/sft-retail-e2e.json').read_text(),ROLLOUT_CONFIG=(ROOT/'configs/rl-retail-tau2-curriculum.json').read_text(),TAU2_SPLIT_MANIFEST=(ROOT/'outputs/retail-data/split_manifest.json').read_text(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',PYTHONUNBUFFERED='1',HF_HUB_DISABLE_PROGRESS_BARS='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,flavor='l40sx1',data_archive_sha256=sha);ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
