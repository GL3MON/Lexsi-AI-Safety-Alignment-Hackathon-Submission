"""Monitor an authorized Kaggle run and download its outputs on completion."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        result = subprocess.run(['kaggle', 'kernels', 'status', args.kernel],
                                capture_output=True, text=True, timeout=90)
        status = (result.stdout + result.stderr).strip()
        record = {'kernel': args.kernel, 'status': status,
                  'checked_at': time.time(), 'exit_code': result.returncode}
        (args.output / 'remote-status.json').write_text(json.dumps(record, indent=2) + '\n')
        print(status, flush=True)
        if result.returncode == 0 and any(word in status.upper()
                                         for word in ['.COMPLETE', '.ERROR', '.FAILED', '.CANCELLED', '.CANCEL_ACKNOWLEDGED']):
            download = subprocess.run(['kaggle', 'kernels', 'output', args.kernel,
                                       '-p', str(args.output), '--force'],
                                      capture_output=True, text=True, timeout=180)
            record.update(download_exit_code=download.returncode,
                          download_message=(download.stdout + download.stderr).strip())
            (args.output / 'remote-status.json').write_text(json.dumps(record, indent=2) + '\n')
            print(record['download_message'], flush=True)
            return
        time.sleep(30)
    (args.output / 'watcher-timeout.json').write_text(json.dumps({
        'kernel': args.kernel, 'watcher_timeout': True,
        'note': 'Monitoring ended; this does not cancel the remote run.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
