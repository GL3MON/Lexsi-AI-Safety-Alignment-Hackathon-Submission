import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records,encode_records
cfg=json.load(open(sys.argv[1]));tok=AutoTokenizer.from_pretrained(cfg['initial_adapter']);tok.chat_template=Path(cfg['chat_template_path']).read_text()
train=read_records(Path(cfg['dataset_path']));validation=read_records(Path(cfg['eval_dataset_path']))
assert len(train)==432 and len(validation)==48
assert not {r['group_id'] for r in train}&{r['group_id'] for r in validation}
n=0;maximum=0;supervised=0
for row,enc,spans,error in encode_records(tok,train+validation,8192,1):
 assert not error,(row['id'],error)
 assert row['loss_message_indices']==[len(row['messages'])-1]
 selected=set();indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
 for index,(start,end) in zip(indices,spans,strict=True):
  if index==len(row['messages'])-1:
   assert enc['labels'][start:end]==enc['input_ids'][start:end];selected.update(range(start,end))
  else:assert all(v==-100 for v in enc['labels'][start:end])
 assert all(v==-100 for i,v in enumerate(enc['labels']) if i not in selected)
 n+=1;maximum=max(maximum,len(enc['input_ids']));supervised+=len(selected)
 if n%100==0:print('AUDITED',n,flush=True)
print(json.dumps(dict(records=n,max_tokens=maximum,supervised_tokens=supervised,one_next_response_supervised=True,other_messages_masked=True)),flush=True)
