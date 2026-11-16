"""Submit a budget-bounded single-epoch Salesforce retail SFT job."""
import argparse,json,netrc
from pathlib import Path
from huggingface_hub import HfApi,get_token

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-minutes',type=int,default=240)
    args=parser.parse_args()
    if not 1<=args.timeout_minutes<=240:raise ValueError('SFT timeout must be at most four hours')
    root=Path(__file__).resolve().parents[2]
    ledger_file=root/'hf_jobs/retail_rl/artifacts/budget-ledger.json'
    ledger=json.loads(ledger_file.read_text());api=HfApi()
    hardware=next(h for h in api.list_jobs_hardware() if h.name=='l40sx1')
    assert hardware.unit_label=='minute'
    rate=hardware.unit_cost_micro_usd/1_000_000;minutes=args.timeout_minutes;reserve=round((minutes+1)*rate+.01,2)
    committed=sum(r.get('settled_maximum_usd',r['reserved_maximum_usd']) for r in ledger)
    if committed+reserve>15:raise RuntimeError('Protected $5 reserve would be consumed')
    auth=netrc.netrc().authenticators('api.forge.coreweave.com');assert auth and auth[2]
    input_repo='GL3MON/tiny-aya-apigen-retail-sft-input';result_repo='GL3MON/tiny-aya-apigen-retail-sft-results'
    for repo in [input_repo,result_repo]:assert api.repo_info(repo,repo_type='dataset').private
    command=['sh','-c','python -m pip install -q huggingface_hub==1.33.0 && python -c "import os,runpy;from huggingface_hub import hf_hub_download;runpy.run_path(hf_hub_download(os.environ[\'SFT_INPUT_REPO\'],\'bootstrap.py\',repo_type=\'dataset\'),run_name=\'__main__\')"']
    job=api.run_job(image='python:3.12-slim-bookworm',command=command,flavor='l40sx1',timeout=minutes*60,
        namespace='GL3MON',name='aya-apigen-retail-sft-1epoch',
        env=dict(SFT_INPUT_REPO=input_repo,SFT_RESULT_REPO=result_repo,HF_HUB_DISABLE_PROGRESS_BARS='1',
                 WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives'),
        secrets=dict(HF_TOKEN=get_token(),WANDB_API_KEY=auth[2]))
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    entry=dict(job_id=job.id,url=job.url,reserved_maximum_usd=reserve,flavor='l40sx1',timeout_minutes=minutes,
               task='apigen-retail-sft',epochs=1,initial_checkpoint='glaive-rl17',batch_size=cfg['batch_size'],
               gradient_accumulation_steps=cfg['gradient_accumulation_steps'],prior_committed_upper_bound_usd=round(committed,2))
    ledger.append(entry);ledger_file.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(entry),flush=True)

if __name__=='__main__':main()
