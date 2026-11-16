"""Independently bind each diagnostic artifact to its intended raw/rendered case."""
import argparse,json
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('evaluation_root',type=Path);args=parser.parse_args();data=Path(__file__).resolve().parents[1]/'outputs/retail-data/clarification-diagnostic20';known={r['id']:r for split in ['train','heldout'] for r in map(json.loads,(data/(split+'.jsonl')).read_text().splitlines())};report={}
 for phase in ['before','after']:
  prompts=[];counts={}
  for split in ['train','heldout']:
   directory=args.evaluation_root/f'short-{phase}-{split}';rows=[json.loads(p.read_text()) for p in directory.glob('*-bank.json')]
   if len(rows)!=40:raise ValueError(f'{phase}/{split}: incomplete: {len(rows)}/40')
   for row in rows:
    expected=known[row['task_id']];action=row['actions'][0];actual_text=expected['messages'][1]['content']
    assert [m['role'] for m in action['messages']]==['system','user'],'Assistant target leaked into prompt'
    assert row['messages'][1]['content']==actual_text
    assert action['messages'][1]['content']==actual_text,(row['task_id'],'rendered case mismatch')
    assert actual_text in action['rendered_prompt']
    prompts.append(tuple(action['prompt_ids']))
   counts[split]=sum(r['short_task_success'] for r in rows)
  assert len(prompts)==len(set(prompts))==80,(phase,'duplicate prompts')
  report[phase]=dict(cases=80,distinct_prompt_ids=80,raw_rendered_identity_checks=True,passed=counts)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
