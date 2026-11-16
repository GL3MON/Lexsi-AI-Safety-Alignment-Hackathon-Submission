import json,re,random,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent/'data';out.mkdir(exist_ok=True)
system=json.loads((root/'outputs/retail-data/apigen-mt-retail-sft/train.jsonl').open().readline())['messages'][0]['content']
summary={}
for split,filename,limit in [('train','sft.jsonl',128),('validation','validation.jsonl',16)]:
 rows=[]
 for line in (root/'outputs/retail-data/retail-sdg-v2-strict-clean'/filename).read_text().splitlines():
  r=json.loads(line);msgs=r['messages'];first=next((i for i,m in enumerate(msgs) if m['role']=='assistant' and '<tool_call>' in m['content']),None)
  if first is None:continue
  ask=[i for i,m in enumerate(msgs[:first]) if m['role']=='assistant' and re.search(r'\?|please|provide|share',m['content'],re.I) and re.search(r'email|\bzip\b|last name|first name',m['content'],re.I)]
  if not ask:continue
  target=[i for i in r['loss_message_indices'] if i<=first]
  assert all(i in target for i in ask) and first in target
  row={k:r[k] for k in ['id','task_id','group_id','split','domain']};row.update(messages=[dict(m) for m in msgs[:first+1]],loss_message_indices=target)
  row['messages'][0]={'role':'system','content':system};rows.append(row)
 random.Random(42).shuffle(rows);rows=rows[:limit]
 (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 summary[split]=len(rows)
 if split=='train':train_ids={r['group_id'] for r in rows}
 else:assert not train_ids & {r['group_id'] for r in rows}
summary.update(source='retail-sdg-v2-strict-clean; original splits preserved',selection='Before first tool: supervised request for identity details, followed by supplied values and first correct tool call',initial_checkpoint=50,system_prompt='Unmodified Salesforce evaluation system; no added anti-invention rule',seed=42)
(out/'manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
