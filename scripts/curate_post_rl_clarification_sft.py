"""Curate verified clarification+retention demonstrations from the training scenario bank."""
import argparse,copy,hashlib,json,random,re
from collections import Counter
from pathlib import Path
from tau2_grpo.rl_bank import load_bank
from tau2_grpo.short_scenario_bank import BankEnvironment,matches
from tau2_grpo.sft_data import tokenize_assistant_only
ROOT=Path(__file__).resolve().parents[1]
CLARIFY=('missing_identity','missing_first_name','missing_last_name','missing_zip','partial_first_name','partial_last_name','partial_zip','failed_email')
EXCLUDED=('cancel_needs_confirmation','confirmation_declined')
QUESTIONS=(
 'Could you please provide your {field}?',
 'What is your {field}?',
 'Please share your {field} so I can look up the account.',
 'Can you tell me your {field}?',
 'To locate the account, please provide your {field}.',
 'Would you please send your {field}?',
 'To continue, could you please share your {field}?' ,
 'Please tell me your {field} before I look up the account.',
 'Can you share your {field} for the account lookup?',
 'What is the {field} associated with your account?',
 'Please provide just your {field}; I can use the details already supplied.',
 'Could you share your {field} so I can continue with the lookup?',
)
NOTES={
 'find_user_id_by_email':"I'll look up the account using the email you provided.",
 'find_user_id_by_name_zip':"I'll look up the account with the name and ZIP code you provided.",
 'get_user_details':"I'll retrieve the account details.",
 'get_order_details':"I'll check the order details.",
 'get_product_details':"I'll retrieve the product options and prices.",
 'list_all_product_types':"I'll check which product types are available.",
 'calculate':"I'll calculate that using the amounts provided.",
 'cancel_pending_order':"I'll cancel the entire order as you confirmed.",
 'modify_pending_order_address':"I'll update the order's shipping address as confirmed.",
 'modify_user_address':"I'll update the account address as confirmed.",
 'modify_pending_order_payment':"I'll change the payment method as confirmed.",
 'modify_pending_order_items':"I'll apply the item changes you confirmed.",
 'return_delivered_order_items':"I'll submit the confirmed return request.",
 'exchange_delivered_order_items':"I'll submit the confirmed exchange request.",
 'transfer_to_human_agents':"I'll transfer this request to a human agent.",
}

def render_action(step,index):
 if step['kind']=='ask':
  label={'email':'email address','first_name':'first name','last_name':'last name','zip':'ZIP code'}[step['field']]
  if step.get('correction'):label='correct '+label
  return QUESTIONS[index%len(QUESTIONS)].format(field=label)
 if step['kind']=='call':
  call='<tool_call>'+json.dumps(dict(name=step['name'],arguments=step['arguments']),separators=(',',':'),ensure_ascii=False)+'</tool_call>'
  return (NOTES[step['name']]+'\n' if index%2 else '')+call
 if step['kind']=='status':return f'Order {step["order_id"]} is currently {step["status"]}.'
 raise ValueError('Unsupported supervised step: '+step['kind'])

def demonstration(case,world,catalog,index):
 route=case['routes'][index%len(case['routes'])]
 selected=copy.deepcopy(case);selected['routes']=[route]
 env=BankEnvironment(selected,world,catalog);messages=env.reset();prefix=len(messages);targets=[];audit=[]
 for n,step in enumerate(route):
  text=render_action(step,index+n)
  incoming,done,call=env.step(text)
  if env.done and not env.success:raise ValueError((case['id'],text,env.checks))
  targets.append(len(messages));messages.append(dict(role='assistant',content=text));messages.extend(copy.deepcopy(incoming));audit.append(dict(kind=step['kind'],field=step.get('field'),name=step.get('name'),passed=env.checks[-1]['passed']))
 reward,sim=env.finish();assert reward==1. and env.success
 # Only the newly constructed successful decisions are targets. Historical calls,
 # including corruptions and failed lookups, stay context-only.
 return dict(id='postrl-clarify-'+case['id'],domain='retail',task_id=case['world_id'],source_case_id=case['id'],source_task_id=case['source_task_id'],world_id=case['world_id'],family=case['family'],category='clarification' if case['family'] in CLARIFY else 'retention',messages=messages,loss_message_indices=targets,prefix_message_count=prefix,verification=dict(executed=True,short_goal_reward=reward,steps=audit,state_hash_checked=True),route_index=index%len(case['routes']))

def audit_labels(row,tokenizer):
 encoded,spans=tokenize_assistant_only(tokenizer,row['messages'],8192,row['loss_message_indices'])
 indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant'];assert len(indices)==len(spans)
 selected=set(row['loss_message_indices']);covered=set();targets=[]
 for i,(start,end) in zip(indices,spans,strict=True):
  values=encoded['labels'][start:end]
  if i in selected:
   assert values==encoded['input_ids'][start:end];covered.update(range(start,end));targets.append(tokenizer.decode(encoded['input_ids'][start:end],skip_special_tokens=False))
  else:assert all(v==-100 for v in values),'Historical assistant context was supervised'
 assert all(label==-100 for i,label in enumerate(encoded['labels']) if i not in covered)
 assert min(selected)>=row['prefix_message_count']
 return dict(id=row['id'],tokens=len(encoded['input_ids']),supervised_tokens=len(covered),assistant_targets=targets,selected_message_indices=row['loss_message_indices'],historical_assistant_masked=True,user_tool_system_masked=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/retail-data/post-rl-clarification200-retention100');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'examples').mkdir(exist_ok=True)
 root,bank,pools,catalog=load_bank('outputs/retail-data/short-scenario-bank-3000');rng=random.Random(42);retention=sorted({r['family'] for r in bank.values()}-set(CLARIFY)-set(EXCLUDED));assert len(retention)==20
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'outputs/rl-initialization/short-bank-checkpoint-28');assert tokenizer.chat_template==(ROOT/'templates/retail_sdg_v2.jinja').read_text()
 datasets={};audits={}
 for split in ('train','validation'):
  selected=[]
  if split=='train':
   for family in CLARIFY+tuple(retention):
    candidates=sorted((bank[c] for c in pools['train'] if bank[c]['family']==family),key=lambda r:r['id']);rng.shuffle(candidates)
    selected.extend(candidates[:25 if family in CLARIFY else 5])
  else:
   # Original fixed greedy world80 and the test partition stay out of SFT.
   for family in CLARIFY:selected.extend(bank[f'world-{w:04d}-{family}'] for w in (81,82,83))
   selected.extend(bank[f'world-0084-{family}'] for family in retention)
  rows=[];checked=[]
  for index,case in enumerate(selected):
   world=json.loads((root/case['world_file']).read_text());row=demonstration(case,world,catalog,index);checked.append(audit_labels(row,tokenizer));rows.append(row)
   if (index+1)%25==0:print(f'VERIFIED {split} {index+1}/{len(selected)}',flush=True)
  rng.shuffle(rows);datasets[split]=rows;audits[split]=checked
  (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  for family in CLARIFY+tuple(retention):
   sample=next(r for r in rows if r['family']==family)
   text='\n\n'.join(f'--- MESSAGE {i}: {m["role"].upper()}'+(' [SFT TARGET]' if i in sample['loss_message_indices'] else ' [MASKED CONTEXT]')+' ---\n'+m['content'] for i,m in enumerate(sample['messages']))
   (out/'examples'/f'{split}-{family}.txt').write_text(text)
 assert len(datasets['train'])==300 and Counter(r['category'] for r in datasets['train'])=={'clarification':200,'retention':100}
 train_worlds={r['world_id'] for r in datasets['train']};val_worlds={r['world_id'] for r in datasets['validation']};assert not train_worlds&val_worlds
 assert all(int(w.split('-')[1])<80 for w in train_worlds)
 assert len({json.dumps(r['messages'],sort_keys=True) for rows in datasets.values() for r in rows})==344
 supervised=Counter()
 for r in datasets['train']:
  for i in r['loss_message_indices']:
   if r['messages'][i]['content'].startswith('<tool_call>') or '<tool_call>' in r['messages'][i]['content']:supervised['tool_calls']+=1
   else:supervised['customer_responses']+=1
 manifest=dict(complete=True,initial_checkpoint='short bank RL checkpoint28',initial_adapter_sha256=hashlib.sha256((ROOT/'outputs/rl-initialization/short-bank-checkpoint-28/adapter_model.safetensors').read_bytes()).hexdigest(),training_records=300,clarification_records=200,retention_records=100,validation_records=len(datasets['validation']),clarification_families=list(CLARIFY),retention_families=retention,train_counts=dict(Counter(r['family'] for r in datasets['train'])),validation_counts=dict(Counter(r['family'] for r in datasets['validation'])),train_worlds=sorted(train_worlds),validation_worlds=sorted(val_worlds),world_overlap=0,original_validation_world80_used_for_sft=False,test_worlds_used=False,source='Executable canonical routes from completed synthetic short-scenario bank; existing Qwen-reworded customer contexts, diverse controlled assistant wording, real RetailTools observations.',loss_mask='Only new successful assistant decisions: clarification and subsequent calls; system, users, tools, all historical assistant messages and errors masked.',max_tokens=max(a['tokens'] for rows in audits.values() for a in rows),verified_records=344,supervised_message_types=dict(supervised),data_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in datasets})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'mask-audit.json').write_text(json.dumps(audits,indent=2,ensure_ascii=False));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
