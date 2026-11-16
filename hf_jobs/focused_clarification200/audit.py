import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records,encode_records
cfg=json.load(open(sys.argv[1]));tok=AutoTokenizer.from_pretrained(cfg['initial_adapter']);tok.chat_template=Path(cfg['chat_template_path']).read_text()
counts=dict(records=0,max_tokens=0,supervised_tokens=0,masked_assistant_turns=0,supervised_assistant_turns=0)
for row,enc,spans,error in encode_records(tok,read_records(Path(cfg['dataset_path']))+read_records(Path(cfg['eval_dataset_path'])),8192,1):
 assert not error,(row['id'],error)
 indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant'];targets=set(row['loss_message_indices']);selected=set()
 assert len(targets)==(1 if row['complete_information'] else 2)
 for index,(start,end) in zip(indices,spans,strict=True):
  if index in targets:
   assert enc['labels'][start:end]==enc['input_ids'][start:end];selected.update(range(start,end));counts['supervised_assistant_turns']+=1
  else:
   assert all(v==-100 for v in enc['labels'][start:end]);counts['masked_assistant_turns']+=1
 assert all(v==-100 for i,v in enumerate(enc['labels']) if i not in selected)
 counts['records']+=1;counts['max_tokens']=max(counts['max_tokens'],len(enc['input_ids']));counts['supervised_tokens']+=len(selected)
 if counts['records']%50==0:print('AUDIT',counts['records'],flush=True)
print(json.dumps(dict(**counts,assistant_only_selected_masks_verified=True)),flush=True)
