"""Save an existing HF Job's live logs and status; never submit paid compute."""
import argparse
import json
import time
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--namespace', default='GL3MON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results-repo',default='GL3MON/tau2-retail-rl-job-results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    deadline = time.monotonic() + 21600
    while time.monotonic() < deadline:
        job = api.inspect_job(job_id=args.job_id, namespace=args.namespace)
        status = {'job_id': args.job_id, 'url': job.url, 'stage': str(job.status.stage),
                  'message': job.status.message, 'checked_at': time.time()}
        try:
            logs = list(api.fetch_job_logs(job_id=args.job_id, namespace=args.namespace, follow=False))
            (args.output / 'job.log').write_text('\n'.join(logs) + '\n')
        except Exception as error:
            status['log_fetch_error'] = type(error).__name__
        (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
        if str(job.status.stage).upper() in {'COMPLETED', 'ERROR', 'FAILED', 'CANCELED', 'CANCELLED', 'TIMEOUT', 'STOPPED'}:
            try:
                snapshot_download(args.results_repo, repo_type='dataset',
                                  local_dir=args.output / 'results',
                                  allow_patterns=['job-reports/*','reports/*','training/**/metrics.json', 'training/**/result.json',
                                                  'training/**/groups.json', 'training/**/runtime.json'])
            except Exception as error:
                (args.output / 'results-download-error.txt').write_text(type(error).__name__ + '\n')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
