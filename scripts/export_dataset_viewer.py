"""Create a standalone browser viewer with an embedded JSONL dataset snapshot."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = '<script id="embedded-data" type="application/json">null</script>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not records or any(not isinstance(r.get('messages'), list) for r in records):
        raise ValueError('Expected JSONL records with messages arrays')
    # Prevent any dataset content from terminating the inert JSON script block.
    data = json.dumps({'name': args.dataset.name, 'records': records}, ensure_ascii=False).replace('<', '\\u003c')
    template = (ROOT/'tools/dataset_viewer.html').read_text()
    assert template.count(MARKER) == 1
    result = template.replace(MARKER, '<script id="embedded-data" type="application/json">'+data+'</script>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f'{args.output.resolve()} ({len(records)} conversations)')


if __name__ == '__main__':
    main()
