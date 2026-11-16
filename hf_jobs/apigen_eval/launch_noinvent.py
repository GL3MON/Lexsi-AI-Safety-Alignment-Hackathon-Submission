import json
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,get_token
root=Path(__file__).resolve().parents[2];api=HfApi()
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
input_repo='GL3MON/aya-apigen-noinvent-eval-input';result_repo='GL3MON/aya-apigen-noinvent-eval-'+stamp
api.create_repo(input_repo,repo_type='dataset',private=True,exist_ok=True)
api.create_repo(result_repo,repo_type='dataset',private=True)
row=json.loads((root/'outputs/retail-data/apigen-mt-retail-sft/train.jsonl').open().readline())
system=row['messages'][0]['content'];assert row['messages'][0]['role']=='system'
system += "\n\nTool argument grounding: Never invent, guess, or fill in missing tool argument values. Use only values explicitly supplied by the customer or returned by a tool. Copy identifiers, names, email addresses, ZIP codes, and order IDs exactly; do not infer them from other fields or use placeholders. If a required value is missing or a lookup fails and you lack the values for an alternative lookup, ask the customer for the missing information before calling a tool. Use only listed tools and their exact argument names."
for name,data in [('bootstrap.py',(root/'hf_jobs/apigen_eval/bootstrap_noinvent.py').read_bytes()),('evaluate.py',(root/'scripts/eval_retail_tool_generalization.py').read_bytes()),('system.txt',system.encode())]:
    api.upload_file(repo_id=input_repo,repo_type='dataset',path_in_repo=name,path_or_fileobj=data)
hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1');assert hardware.unit_label=='minute'
reserve=round(31*hardware.unit_cost_micro_usd/1e6+.01,2)
ledger_file=root/'hf_jobs/retail_rl/artifacts/budget-ledger.json';ledger=json.loads(ledger_file.read_text())
committed=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger)
if committed+reserve>15:raise RuntimeError(f'Protected reserve: committed={committed}, new={reserve}')
command=['sh','-c','pip install -q huggingface_hub==1.33.0 && python -c "import os,runpy;from huggingface_hub import hf_hub_download;runpy.run_path(hf_hub_download(os.environ[\'EVAL_INPUT\'],\'bootstrap.py\',repo_type=\'dataset\'),run_name=\'__main__\')"']
job=api.run_job(image='pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime',command=command,flavor='l40sx1',timeout=1800,namespace='GL3MON',name='aya-apigen-noinvent-17-50-96',env={'EVAL_INPUT':input_repo,'EVAL_RESULTS':result_repo,'HF_HUB_DISABLE_PROGRESS_BARS':'1','PYTHONUNBUFFERED':'1','PIP_BREAK_SYSTEM_PACKAGES':'1'},secrets={'HF_TOKEN':get_token()})
entry=dict(job_id=job.id,url=job.url,reserved_maximum_usd=reserve,task='noinvent-inference-17-50-96',results_repo=result_repo,timeout_minutes=30)
ledger.append(entry);ledger_file.write_text(json.dumps(ledger,indent=2)+'\n')
(root/'hf_jobs/apigen_eval/job-noinvent.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
