import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
source=ROOT/'hf_jobs/targeted_clarification_sft200/data'
for filename in ['train.jsonl','validation.jsonl']:
 rows=[json.loads(l) for l in (source/filename).read_text().splitlines()]
 for row in rows:
  selected=[]
  for i,m in enumerate(row['messages']):
   if m['role']!='assistant' or '<tool_call>' not in m['content']:continue
   call=json.loads(m['content'].split('<tool_call>',1)[1].split('</tool_call>',1)[0])
   if not call['name'].startswith('find_user_id_'):continue
   result=json.loads(row['messages'][i+1]['content'])['content']
   if isinstance(result,str) and result.startswith('Error:'):continue
   if not row['complete_information']:
    question=max(j for j in range(i) if row['messages'][j]['role']=='assistant' and '<tool_call>' not in row['messages'][j]['content'])
    selected.append(question)
   selected.append(i);break
  assert len(selected)==(1 if row['complete_information'] else 2)
  row['loss_message_indices']=selected
 (HERE/'data'/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
manifest=json.loads((source/'manifest.json').read_text());manifest.update(experiment='Only loss_message_indices changed; identical 200/24 messages, ordering, split and source checkpoint50',supervision='Missing cases: final specific clarification and subsequent successful identity lookup. Complete cases: immediate successful identity lookup. All other assistant turns, users and tools masked.',data_sha256={f:hashlib.sha256((HERE/'data'/f).read_bytes()).hexdigest() for f in ['train.jsonl','validation.jsonl']})
(HERE/'data/manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared focused labels for 200 train / 24 validation')
