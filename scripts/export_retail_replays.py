"""Export saved retail actor conversations without rerunning inference."""

import argparse
import json
from pathlib import Path

from tau2_grpo.rl_reporting import write_episode_conversation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    files = sorted(args.run.glob('group-*/episode-*.json'))
    files += [args.run / (name + '.json') for name in ('function_calling_base', 'retail_sft')
              if (args.run / (name + '.json')).is_file()]
    for file in files:
        print(write_episode_conversation(file, json.loads(file.read_text())))


if __name__ == '__main__':
    main()
