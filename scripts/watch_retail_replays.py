"""Refresh readable replay files while a known local actor process is running."""

import argparse
import json
import time
from pathlib import Path

from tau2_grpo.rl_reporting import write_episode_conversation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    while True:
        command = Path(f'/proc/{args.pid}/cmdline')
        running = command.exists() and b'tau2_grpo.rl' in command.read_bytes()
        progress = [f'Actor process running: {running}', '']
        for source in sorted(args.run.glob('group-*/episode-*.json')):
            episode = json.loads(source.read_text())
            write_episode_conversation(source, episode)
            progress.append(f"{source.stem}: seed={episode['seed']}, actor_turns={len(episode['actions'])}, complete={episode['complete']}, reward={episode['reward']}")
        (args.run/'progress.txt').write_text('\n'.join(progress)+'\n')
        if not running:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
