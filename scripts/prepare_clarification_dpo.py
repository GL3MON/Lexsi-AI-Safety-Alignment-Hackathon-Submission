"""Grounded DPO decision pairs with exactly shared prompt and verified preferences."""
import copy,json,hashlib,random
from pathlib import Path
from collections import Counter
from tau2_grpo.rl_bank import load_bank
from tau2_grpo.short_scenario_bank import BankEnvironment
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.rl_context import student_messages
ROOT=Path(__file__).resolve().parents[1]

from tau2_grpo.dpo_data import render_completion

def verify_pair(case,world,catalog,chosen,rejected):
 result=[]
 for text in [chosen,rejected]:
  env=BankEnvironment(case,world,catalog);env.reset();env.step(text);result.append(env.checks[-1]['passed']);env.close()
 if result!=[True,False]:raise ValueError((case['id'],result,chosen,rejected))

def bad_lookup(step):
 args=copy.deepcopy(step['arguments']);name=step['name']
 if name=='find_user_id_by_email':args['email']='unknown.customer@example.org'
 else:
  # Use observed recurrent failure forms as explicit rejected candidates.
  for field,value in [('zip','12345'),('last_name','Doe'),('first_name','John')]:
   if field in args:args[field]=value;break
 return '<tool_call>'+json.dumps(dict(name=name,arguments=args),separators=(',',':'))+'</tool_call>'

def main():
 root,bank,pools,catalog=load_bank('outputs/retail-data/short-scenario-bank-3000')
 source=ROOT/'outputs/retail-data/post-rl-clarification200-retention100';out=ROOT/'outputs/retail-data/clarification-dpo';out.mkdir(parents=True,exist_ok=True);(out/'examples').mkdir(exist_ok=True)
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19')
 sets={}
 for split in ['train','validation']:
  original=[json.loads(l) for l in (source/(split+'.jsonl')).read_text().splitlines()];clar=[r for r in original if r['category']=='clarification'];ret=[r for r in original if r['category']=='retention']
  if split=='train':
   # Balanced 12/13 per missing-field family; no held-out-world generations used.
   families=sorted({r['family'] for r in clar});selected=[]
   for i,family in enumerate(families):
    pool=[r for r in clar if r['family']==family];random.Random(42+i).shuffle(pool);selected.extend(pool[:13 if i<4 else 12])
   clar=selected;assert len(clar)==100
  rows=[]
  for r in clar:
   case=copy.deepcopy(bank[r['source_case_id']]);world=json.loads((root/case['world_file']).read_text());route=case['routes'][r['route_index']];assert route[0]['kind']=='ask'
   question=r['messages'][r['loss_message_indices'][0]]['content'];call_step=route[1]
   if case['family']!='failed_email':
    args=copy.deepcopy(call_step['arguments']);field=route[0]['field'];key='email' if field=='email' else field
    args[key]={'email':'unknown.customer@example.org','first_name':'John','last_name':'Doe','zip':'12345'}[key]
    rejected='<tool_call>'+json.dumps(dict(name=call_step['name'],arguments=args),separators=(',',':'))+'</tool_call>'
   else:rejected=bad_lookup(call_step)
   verify_pair(case,world,catalog,question,rejected)
   rows.append(dict(id=r['id']+'-ask-pair',category='missing_ask',family=r['family'],world_id=r['world_id'],prompt=case['messages'],chosen=question,rejected=rejected,rejected_source='controlled reconstruction of observed unsupported-argument failure',verification='Chosen next action passes; rejected next action fails in real isolated environment'))
   complete=copy.deepcopy(case);complete['routes']=[route[1:]]
   if case['family']=='failed_email':
    # Actual failure remains visible; the customer then supplies a correction.
    complete['messages'].append(dict(role='assistant',content='The lookup did not match an account.'))
    complete['messages'].append(dict(role='user',content=route[0]['reply']))
   else:
    assert complete['messages'][-1]['role']=='user'
    complete['messages'][-1]['content']+=' '+route[0]['reply']
   chosen='<tool_call>'+json.dumps(dict(name=call_step['name'],arguments=call_step['arguments']),separators=(',',':'))+'</tool_call>'
   verify_pair(complete,world,catalog,chosen,question)
   rows.append(dict(id=r['id']+'-supplied-pair',category='supplied_call',family=r['family'],world_id=r['world_id'],prompt=complete['messages'],chosen=chosen,rejected=question,rejected_source='Matched unnecessary clarification after required field supplied',verification='Chosen actual lookup passes; repeated question fails next-action check'))
  for r in ret:
   case=copy.deepcopy(bank[r['source_case_id']]);world=json.loads((root/case['world_file']).read_text());first=r['loss_message_indices'][0];chosen=r['messages'][first]['content'];rejected='Could you please provide your order ID?' if case['family'] in ('calculate_total','calculate_difference','get_order','order_status','recover_malformed_json','recover_unknown_tool','recover_wrong_argument') else 'Could you please provide your email address?'
   verify_pair(case,world,catalog,chosen,rejected)
   rows.append(dict(id=r['id']+'-retention-pair',category='retention',family=r['family'],world_id=r['world_id'],prompt=case['messages'],chosen=chosen,rejected=rejected,rejected_source='Controlled unnecessary clarification instead of supported next action',verification='Chosen next action passes; rejected unnecessary question fails'))
  for row in rows:
   p,c=render_completion(tokenizer,row['prompt'],row['chosen']);p2,r=render_completion(tokenizer,row['prompt'],row['rejected']);assert p==p2 and c!=r
   row.update(prompt_ids=p,chosen_ids=c,rejected_ids=r,prompt_sha256=hashlib.sha256(json.dumps(p).encode()).hexdigest(),native_tau2_task=False)
  assert len({(r['prompt_sha256'],r['chosen'],r['rejected']) for r in rows})==len(rows)
  random.Random(42).shuffle(rows);sets[split]=rows
  (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  for category in ['missing_ask','supplied_call','retention']:
   row=next(r for r in rows if r['category']==category);text='\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in row['prompt'])+'\n\nCHOSEN:\n'+row['chosen']+'\n\nREJECTED:\n'+row['rejected'];(out/'examples'/f'{split}-{category}.txt').write_text(text)
 trainworlds={r['world_id'] for r in sets['train']};valworlds={r['world_id'] for r in sets['validation']};assert not trainworlds&valworlds;assert all(int(w.split('-')[1])<80 for w in trainworlds)
 manifest=dict(complete=True,records={s:len(r) for s,r in sets.items()},categories={s:dict(Counter(r['category'] for r in rows)) for s,rows in sets.items()},world_overlap=0,initializer='SFT19',initial_adapter_sha256='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811',max_tokens=max(len(r['prompt_ids'])+max(len(r['chosen_ids']),len(r['rejected_ids'])) for rows in sets.values() for r in rows),same_prompt_verified=True,chosen_rejected_decision_checks_verified=True,eos='Exact rendered assistant suffix, trimmed at first EOS inclusive to match rollout stopping.',data_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in sets},note='Controlled decision-level preference pairs; rejected candidates are reconstructed from observed failure types, not necessarily verbatim student samples. Verification is a next-action rubric, not native Tau2 task success.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
