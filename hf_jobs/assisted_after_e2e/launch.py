import argparse,ast,base64,hashlib,json,netrc,zlib
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token,hf_hub_download
from huggingface_hub._space_api import Volume
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--latest-sft-checkpoint',action='store_true');p.add_argument('--resume-smoke-prefix');a=p.parse_args();api=HfApi();account=api.whoami()['name'];original=json.loads((ROOT/'hf_jobs/clarification_diagnostic/job.json').read_text());source=json.loads((ROOT/'hf_jobs/clarification_dpo/job.json').read_text());assert account==original['account']==source['account']
 hf_hub_download('GL3MON/tiny-aya-global-function-calling-sft','config.json',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',force_download=True)
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text());previous=sum(e.get('settled_maximum_usd',e['reserved_maximum_usd']) for e in ledger if e.get('account')==account and e.get('reservation_key')!='retail-e2e-followup');h=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert h.unit_label=='minute';reserve=round(121*h.unit_cost_micro_usd/1e6+.01,2)
 if previous+reserve>20:raise RuntimeError('Account budget exceeded')
 bootstrap=(HERE/'bootstrap.py').read_text();evaluator=json.dumps({f.name:f.read_text() for f in (ROOT/'src/tau2_grpo').glob('*.py')});ast.parse(bootstrap);json.loads(evaluator)
 parent=json.loads((ROOT/'hf_jobs/retail_e2e_sft/job.json').read_text());assert parent['account']==account
 parent_info=api.inspect_job(job_id=parent['job_id'],namespace=account)
 entries=list(api.list_bucket_tree(parent['checkpoint_bucket'],prefix=parent['bucket_prefix'],recursive=True));paths={e.path for e in entries};artifacts=HERE/'parent-artifacts';artifacts.mkdir(exist_ok=True)
 checkpoint_step=None
 if a.latest_sft_checkpoint:
  if parent_info.status.stage not in ('COMPLETED','ERROR','TIMEOUT','CANCELED','STOPPED'):raise RuntimeError('SFT is still active; refusing checkpoint recovery')
  import re
  candidates=[]
  for name in paths:
   match=re.search(r'/checkpoint-(\d+)/trainer_state.json$',name)
   if match:
    prefix_path=name.rsplit('/',1)[0]
    if {prefix_path+'/adapter_model.safetensors',prefix_path+'/adapter_config.json'}<=paths:candidates.append((int(match[1]),prefix_path))
  if not candidates:raise RuntimeError('No saved adapter checkpoint with trainer state')
  checkpoint_step,remote=max(candidates);local=ROOT/'outputs/rl-initialization'/f'retail-e2e-checkpoint-{checkpoint_step}';local.mkdir(parents=True,exist_ok=True)
  names=[e.path for e in entries if e.path.startswith(remote+'/') and e.path.rsplit('/',1)[-1] in ['adapter_model.safetensors','adapter_config.json','trainer_state.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']]
  api.download_bucket_files(parent['checkpoint_bucket'],[(name,local/name.rsplit('/',1)[-1]) for name in names])
  state=json.loads((local/'trainer_state.json').read_text());assert state['global_step']==checkpoint_step
  from safetensors import safe_open
  with safe_open(str(local/'adapter_model.safetensors'),framework='pt',device='cpu') as weights:
   keys=list(weights.keys());assert keys
   for key in keys:assert all(n>0 for n in weights.get_slice(key).get_shape())
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(local);assert tok.eos_token_id==6;assert tok.chat_template==(ROOT/'templates/retail_sdg_v2.jinja').read_text()
  initial_path='/checkpoints/'+remote;initial_sha=hashlib.sha256((local/'adapter_model.safetensors').read_bytes()).hexdigest()
  (local/'provenance.json').write_text(json.dumps(dict(parent_job=parent['job_id'],bucket=parent['checkpoint_bucket'],path=remote,global_step=checkpoint_step,best_eval_loss=state.get('best_metric'),sha256=initial_sha,explicit_user_recovery=True),indent=2)+'\n')
 else:
  if parent_info.status.stage!='COMPLETED':raise RuntimeError('Wait until SFT job and rollout exit')
  marker=parent['bucket_prefix']+'/JOB_FINISHED';assert marker in paths,'SFT or post-training rollout did not finish'
  api.download_bucket_files(parent['checkpoint_bucket'],[(parent['bucket_prefix']+'/selection.json',artifacts/'selection.json'),(parent['bucket_prefix']+'/tau2-test/summary.json',artifacts/'rollout-summary.json')])
  selection=json.loads((artifacts/'selection.json').read_text());rollout=json.loads((artifacts/'rollout-summary.json').read_text());assert rollout['complete'],'Unscored rollout blocks automatic training'
  initial_path=selection['final_adapter'];initial_sha=selection['sha256'];assert initial_path.startswith('/checkpoints/'+parent['bucket_prefix']+'/training/') and initial_path.endswith('/final_adapter')
 prefix='assisted-after-e2e-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 plan=dict(account=account,task='tau2-assisted-after-e2e-sft',inference_only=False,groups=30,rollouts_per_group=16,curriculum='read-only finish assistance, authentication-only assistance, then unassisted; success-gated',assisted_adaptive=True,advancement_threshold=.5,advancement_window_groups=4,easier_assistance_replay=.2,checkpoint=f'End-to-end SFT checkpoint-{checkpoint_step}' if checkpoint_step else 'Selected end-to-end SFT adapter',manual_checkpoint_recovery=a.latest_sft_checkpoint,checkpoint_step=checkpoint_step,resume_smoke_prefix=a.resume_smoke_prefix,parent_sft_job=parent['job_id'],initial_adapter_sha256=initial_sha,real_tau2_rewards=True,grounded_call_bonus_enabled=False,argument_reward_weight=0.,reward_definition='Sum evaluated Tau2 checks minus argument and failure-streak penalties',smoke_rollouts=16,validation_cases=2,validation_every_groups=10,temperature=.7,top_p=1.,response_tokens=256,max_actions=10,baseline='Native Tau2 before/after validation; additive reward logged separately',timeout_minutes=120,reserved_maximum_usd=reserve,remaining_account_budget_usd=round(20-previous-reserve,2),checkpoint_bucket=original['checkpoint_bucket'],bucket_prefix=prefix,hub_output_uploads=False)
 if not a.submit:print(json.dumps(plan,indent=2));return
 if (HERE/'job.json').exists():
  old=json.loads((HERE/'job.json').read_text());stage=api.inspect_job(job_id=old['job_id'],namespace=account).status.stage
  if stage not in ('COMPLETED','ERROR','CANCELED','TIMEOUT','STOPPED'):raise RuntimeError('Assisted Tau2 RL already active')
 overlay=base64.b64encode(zlib.compress(evaluator.encode(),9)).decode();overlay_env={f'SOURCE_OVERLAY_{i//48000}':overlay[i:i+48000] for i in range(0,len(overlay),48000)}
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'EVAL_BOOTSTRAP\'])),\'<retail90-eval>\',\'exec\'),{\'__name__\':\'__main__\'})"'];auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=7200,namespace=account,name='assisted-rl-after-end-to-end-sft',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False),Volume(type='bucket',source=source['checkpoint_bucket'],mount_path='/source',read_only=True)],env=dict(RESUME_SMOKE_PREFIX=a.resume_smoke_prefix or '',INITIAL_ADAPTER_SHA256=initial_sha,INITIAL_ADAPTER_PATH=initial_path,INITIAL_CHECKPOINT_STEP=str(checkpoint_step or ''),SFT_COMPLETE_MARKER='/checkpoints/'+parent['bucket_prefix']+'/SFT_COMPLETE',TAU2_SPLIT_MANIFEST=(ROOT/'outputs/retail-data/split_manifest.json').read_text(),TAU2_CURRICULUM=(HERE/'curriculum.json').read_text(),EVAL_OUTPUT='/checkpoints/'+prefix,SOURCE_INPUT_PREFIX=source['input_prefix'],SOURCE_SHA256=source['source_sha256'],EVAL_CONFIG=(ROOT/'configs/rl-retail-tau2-assisted.json').read_text(),SOURCE_OVERLAY_PARTS=str(len(overlay_env)),**overlay_env,EVAL_BOOTSTRAP=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,flavor='l40sx1');ledger=[e for e in ledger if e.get('reservation_key')!='retail-e2e-followup'];ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
