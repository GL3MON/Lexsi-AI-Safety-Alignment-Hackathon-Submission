import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records,encode_records
cfg=json.load(open(sys.argv[1]));tok=AutoTokenizer.from_pretrained(cfg['initial_adapter']);tok.chat_template=Path(cfg['chat_template_path']).read_text();n=0;max_tokens=0
for row,enc,spans,error in encode_records(tok,read_records(Path(cfg['dataset_path']))+read_records(Path(cfg['eval_dataset_path'])),8192,1):
 assert not error,(row['id'],error)
 indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant'];assert set(indices)==set(row['loss_message_indices']);selected=set()
 for start,end in spans:
  assert enc['labels'][start:end]==enc['input_ids'][start:end];selected.update(range(start,end))
 assert all(v==-100 for i,v in enumerate(enc['labels']) if i not in selected)
 n+=1;max_tokens=max(max_tokens,len(enc['input_ids']))
print(json.dumps(dict(records=n,max_tokens=max_tokens,assistant_only_masks_verified=True)))
