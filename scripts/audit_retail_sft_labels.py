"""Independently check SFT labels using Jinja generation masks and real TRL collator."""
import argparse,concurrent.futures,json,os
from pathlib import Path
TOKENIZER=None;AUDIT_TEMPLATE=None;COLLATOR=None

def initialize(checkpoint,chat_template=None):
 global TOKENIZER,AUDIT_TEMPLATE,COLLATOR
 import torch
 torch.set_num_threads(1)
 from transformers import AutoTokenizer
 from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
 TOKENIZER=AutoTokenizer.from_pretrained(checkpoint)
 if chat_template:TOKENIZER.chat_template=Path(chat_template).read_text()
 template=TOKENIZER.chat_template
 needle="<|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"
 replacement="<|CHATBOT_TOKEN|><|START_RESPONSE|>{% if message.get('_audit_supervised', False) %}{% generation %}{{ message['content'] }}<|END_RESPONSE|>{% endgeneration %}{% else %}{{ message['content'] }}<|END_RESPONSE|>{% endif %}"
 assert template.count(needle)==1
 template=template.replace(needle,replacement)
 end="{% endif %}<|END_OF_TURN_TOKEN|>{% endfor %}"
 new="{% endif %}{% if role in ['assistant','chatbot'] and message.get('_audit_supervised', False) %}{% generation %}<|END_OF_TURN_TOKEN|>{% endgeneration %}{% else %}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}"
 assert template.count(end)==1
 AUDIT_TEMPLATE=template.replace(end,new)
 COLLATOR=DataCollatorForLanguageModeling(pad_token_id=TOKENIZER.pad_token_id)

def check(item):
 from tau2_grpo.sft_data import tokenize_assistant_only
 split,r=item;chosen=set(r['loss_message_indices'])
 actual,spans=tokenize_assistant_only(TOKENIZER,r['messages'],8192,r['loss_message_indices'])
 assert len(actual['input_ids'])==r.get('context_tokens',len(actual['input_ids'])),r['id']+': stored context length disagrees with selected template'
 annotated=[{**m,'_audit_supervised':i in chosen} for i,m in enumerate(r['messages'])]
 independent=TOKENIZER.apply_chat_template(annotated,chat_template=AUDIT_TEMPLATE,tokenize=True,return_dict=True,return_assistant_tokens_mask=True)
 assert actual['input_ids']==independent['input_ids'],r['id']+': independent renderer changed IDs'
 expected=[token if mask else -100 for token,mask in zip(independent['input_ids'],independent['assistant_masks'],strict=True)]
 mismatches=[i for i,(a,b) in enumerate(zip(actual['labels'],expected,strict=True)) if a!=b]
 if mismatches:return {'id':r['id'],'split':split,'passed':False,'mismatches':len(mismatches),'first_mismatches':mismatches[:10]}
 batch=COLLATOR([actual]);assert batch['labels'][0].tolist()==expected,r['id']+': collator changed labels'
 assert batch['input_ids'][0].tolist()==actual['input_ids']
 assistants=[i for i,m in enumerate(r['messages']) if m['role']=='assistant']
 targets=[TOKENIZER.decode(actual['input_ids'][start:end],skip_special_tokens=False) for idx,(start,end) in zip(assistants,spans,strict=True) if idx in chosen]
 return {'id':r['id'],'split':split,'passed':True,'tokens':len(expected),'supervised_tokens':sum(x!=-100 for x in expected),'masked_assistant_turns':len(assistants)-len(chosen),'tool_turns':sum(m['role']=='tool' for m in r['messages']),'targets':targets if r.get('_show_targets') else None}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--tokenizer');p.add_argument('--chat-template');a=p.parse_args()
 run=Path(a.run);cfg=json.loads((run/'config.json').read_text());items=[]
 for split,path in [('train',cfg['dataset_path']),('eval',cfg['eval_dataset_path'])]:
  rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
  if a.limit:rows=rows[:a.limit]
  for i,r in enumerate(rows):r['_show_targets']=i<2;items.append((split,r))
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);results=[]
 if a.workers==1:
  initialize(a.tokenizer or str(run/'checkpoint-50'),a.chat_template);iterator=map(check,items)
  for result in iterator:results.append(result)
 else:
  import multiprocessing
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.tokenizer or str(run/'checkpoint-50'),a.chat_template)) as pool:
   for i,result in enumerate(pool.map(check,items,chunksize=8),1):
    results.append(result)
    if i%25==0:print('Audited',i,'/',len(items),flush=True)
 report={'records_checked':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']],'independent_method':'HF Jinja generation masks; same rendered IDs; real installed TRL collator','results':results}
 (out/'independent-label-audit.json').write_text(json.dumps(report,indent=2))
 (out/'decoded-training-targets.txt').write_text('\n\n'.join(r['id']+' ['+r['split']+']\n'+'\n\n'.join(r['targets']) for r in results if r.get('targets')))
 print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
 if report['failures']:raise SystemExit(1)
if __name__=='__main__':main()
