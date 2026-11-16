"""Explicitly authorized 90-minute retail RL run; no periodic Hub output commits."""
import argparse,base64,json,math,netrc,zlib,ast
from pathlib import Path
from huggingface_hub import HfApi,get_token
from huggingface_hub._space_api import Volume
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def encoded(value):return base64.b64encode(zlib.compress(value,9)).decode()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true');args=parser.parse_args();api=HfApi()
 ledger_path=ROOT/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_path.read_text());hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute';rate=hardware.unit_cost_micro_usd/1e6
 previous=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger);maximum=round(91*rate+.01,2)
 if previous+maximum>20:raise RuntimeError('Would exceed original $20 authorization')
 files={}
 for name in ['rl.py','rl_policy.py','rl_objective.py','rl_transformers_workers.py']:
  rel='src/tau2_grpo/'+name;text=(ROOT/rel).read_text();ast.parse(text,filename=rel);files[rel]=text
 source=(HERE/'bootstrap.py').read_text()
 needle="    project = WORK / 'project'"
 overlay="""
    import base64,zlib
    overlay=json.loads(zlib.decompress(base64.b64decode(os.environ['RL_SOURCE_OVERLAY'])))
    for relative,text in overlay.items():
        if not relative.startswith('src/tau2_grpo/') or '..' in Path(relative).parts:raise RuntimeError('Invalid overlay path')
        (project/relative).write_text(text)
"""
 source=source.replace(needle,needle+overlay)
 source=source.replace("    config = json.loads((project / 'hf-smoke-config.json').read_text())", "    config = json.loads((project / 'hf-smoke-config.json').read_text())\n    config['train_steps']=22")
 ast.parse(source);(HERE/'artifacts/long-bootstrap.py').write_text(source)
 plan=dict(initial_checkpoint='run-2190c37f90/checkpoint-25',groups=22,rollouts_per_group=8,max_actions=10,timeout_minutes=90,maximum_reserved_usd=maximum,prior_committed_upper_usd=round(previous,2),remaining_original_budget_usd=round(20-previous-maximum,2),hub_output_uploads=False,checkpoint_bucket='GL3MON/tau2-retail-rl-checkpoints',checkpoint_retention=2,mlp_chunk_size_on_l40s=1024,authorization='User explicitly requested a 1.5-hour job; original $20 total remains the ceiling')
 if not args.submit:print(json.dumps(plan,indent=2));return
 auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2] and get_token()
 api.create_bucket(plan['checkpoint_bucket'],private=True,exist_ok=True)
 command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import base64,zlib,os;exec(compile(zlib.decompress(base64.b64decode(os.environ[\'RL_BOOTSTRAP_CODE\'])),\'<rl-bootstrap>\',\'exec\'),{\'__name__\':\'__main__\'})"']
 job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=5400,namespace='GL3MON',name='tau2-retail-focused25-grpo-90min',volumes=[Volume(type='bucket',source=plan['checkpoint_bucket'],mount_path='/checkpoints',read_only=False)],env=dict(RL_INPUT_REPO='GL3MON/tau2-retail-focused25-rl-input',RL_RESULT_REPO='GL3MON/tau2-retail-focused25-rl-results',RL_JOB_MODE='train',RL_UPLOAD_OUTPUTS='0',RL_CHECKPOINT_DIR='/checkpoints',RL_BOOTSTRAP_CODE=encoded(source.encode()),RL_SOURCE_OVERLAY=encoded(json.dumps(files).encode()),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1'),secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
 entry=dict(plan,job_id=job.id,url=job.url,reserved_maximum_usd=maximum,task='retail-grpo-90min',flavor='l40sx1')
 ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');(HERE/'job-long.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
if __name__=='__main__':main()
