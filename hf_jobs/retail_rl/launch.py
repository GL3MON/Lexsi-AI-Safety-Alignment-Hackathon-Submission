"""Print a bounded HF Job plan; paid compute requires explicit --submit."""
import argparse
import json
import netrc
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_REPO = 'GL3MON/tau2-retail-rl-job-input'
RESULT_REPO = 'GL3MON/tau2-retail-rl-job-results'


def main():
    from huggingface_hub import HfApi, get_token
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true', help='Actually submit paid compute')
    parser.add_argument('--mode', choices=['train','greedy','rollout-smoke'], default='train')
    parser.add_argument('--flavor', default='l40sx1')
    parser.add_argument('--timeout-minutes', type=int, default=45)
    parser.add_argument('--budget-usd', type=float, default=20.)
    parser.add_argument('--reserve-usd', type=float, default=5.)
    args = parser.parse_args()
    if not 1 <= args.timeout_minutes <= 60:
        raise ValueError('The initial smoke must have a timeout of 1–60 minutes')
    if not 0 < args.budget_usd <= 20:
        raise ValueError('This setup has a hard cumulative $20 authorization ceiling')
    if not 5 <= args.reserve_usd < args.budget_usd:
        raise ValueError('Keep at least $5 uncommitted within the authorized budget')
    spendable = args.budget_usd - args.reserve_usd
    api = HfApi()
    hardware = next(h for h in api.list_jobs_hardware() if h.name == args.flavor)
    rate = hardware.unit_cost_micro_usd / 1_000_000
    if hardware.unit_label != 'minute':
        raise ValueError('Unexpected pricing unit; review pricing before submission')
    artifact_dir = HERE / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    ledger_file = artifact_dir / 'budget-ledger.json'
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    # Reserve rounded-up cents, including a billable-minute margin.
    maximum = round((args.timeout_minutes + 1) * rate + .01, 2)
    previous = sum(row['reserved_maximum_usd'] for row in ledger)
    if previous + maximum > spendable:
        raise RuntimeError('The cumulative spending reservation would consume the protected reserve')
    plan = {'flavor': args.flavor, 'price_per_hour_usd': rate * 60,
            'timeout_minutes': args.timeout_minutes, 'maximum_reserved_usd': maximum,
            'previous_reserved_usd': previous, 'cumulative_budget_usd': args.budget_usd,
            'protected_reserve_usd': args.reserve_usd, 'spendable_budget_usd': spendable,
            'input_repo': INPUT_REPO, 'results_repo': RESULT_REPO,
            'initial_checkpoint': 100, 'rollout_groups': 2, 'rollouts_per_group': 8,
            'max_actions': 10, 'wandb': 'online', 'automatic_retries': False,
            'submitted': False}
    config = json.loads((HERE / 'smoke-config.json').read_text())
    plan.update(rollout_engine=config['rollout_engine'],rollout_workers=config['rollout_workers'],
                rollout_batch_size=config.get('rollout_batch_size',1))
    if args.mode == 'greedy':
        plan.update(mode='greedy',training=False,temperature=0,rollout_groups=0,rollouts_per_group=0)
    elif args.mode == 'rollout-smoke':
        plan.update(mode='rollout-smoke',training=False,temperature=0,rollout_groups=1)
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print('DRY RUN ONLY. No paid compute submitted.')
        return
    for repo in [INPUT_REPO, RESULT_REPO]:
        if not api.repo_info(repo, repo_type='dataset').private:
            raise RuntimeError('Expected private input/results repositories')
    if 'payload.tar.gz' not in api.list_repo_files(INPUT_REPO, repo_type='dataset'):
        raise RuntimeError('Upload the prepared payload first')
    auth = netrc.netrc().authenticators('api.forge.coreweave.com')
    if not auth or not auth[2] or not get_token():
        raise RuntimeError('HF and W&B credentials must be configured before submission')
    # Secrets go through the Jobs secret API, never source files or command arguments.
    bootstrap_file = 'greedy_bootstrap.py' if args.mode == 'greedy' else 'bootstrap.py'
    command = ['sh', '-c',
        'python -m pip install --no-cache-dir huggingface_hub==1.33.0 && '
        'python -c "import os,runpy;from huggingface_hub import hf_hub_download;'
        f'runpy.run_path(hf_hub_download(os.environ[\'RL_INPUT_REPO\'],\'{bootstrap_file}\',repo_type=\'dataset\'),run_name=\'__main__\')"']
    job = api.run_job(image='python:3.12-slim-bookworm', command=command,
        flavor=args.flavor, timeout=args.timeout_minutes * 60, namespace='GL3MON',
        name='tau2-checkpoint100-greedy-check' if args.mode=='greedy' else 'tau2-retail-additive-rl-smoke',
        env={'RL_INPUT_REPO': INPUT_REPO, 'RL_RESULT_REPO': RESULT_REPO,
             'RL_JOB_MODE': args.mode,
             'WANDB_BASE_URL': 'https://api.forge.coreweave.com',
             'WANDB_ENTITY': 'harisshragavarchives'},
        secrets={'HF_TOKEN': get_token(), 'WANDB_API_KEY': auth[2]})
    ledger.append({'job_id': job.id, 'url': job.url,
                   'reserved_maximum_usd': maximum, 'flavor': args.flavor,
                   'timeout_minutes': args.timeout_minutes})
    ledger_file.write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'submitted': True, 'job_id': job.id, 'url': job.url}))


if __name__ == '__main__':
    main()
