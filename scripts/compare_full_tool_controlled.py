"""Isolate copying and workflow decisions with Tau2's full 17-tool prompt."""
import gc,json,os,re,time
from collections import defaultdict
from pathlib import Path
import torch
from jsonschema import validate
from tau2_grpo.rl_config import RetailRLConfig,ROOT
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.prepare_function_calling import parse_calls
from scripts.generate_copy_repair_sft import scenario

OUT=ROOT/'outputs/diagnostics/full-17-tool-controlled-comparison'
MODELS=[('hermes-sft',None),('repair-sft','outputs/sft-retail-copy-repair-500/run-fd7056d70b/checkpoint-100'),('long-sft','outputs/sft-retail-long-500-after-copy-repair/run-cbf90bb883/checkpoint-75')]

def atomic(p,value):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def build_cases():
 from tau2.domains.retail.data_model import get_db
 base=get_db();cases=[]
 def call(name,arguments):return {'role':'assistant','content':"I'll check that now.\n<tool_call>"+json.dumps({'name':name,'arguments':arguments})+'</tool_call>'}
 def obs(name,result):
  if hasattr(result,'model_dump'):result=result.model_dump(mode='json')
  return {'role':'user','content':'<tool_response>'+json.dumps({'name':name,'content':result},ensure_ascii=False)+'</tool_response>'}
 def add(id,category,messages,expected=None,question=None):cases.append({'id':id,'category':category,'messages':messages,'expected_call':expected,'question':question})
 add('missing_all_identity','clarification',[{'role':'user','content':'I need to cancel an order I no longer need. I cannot find the order number.'}],question='identity')
 add('only_zip_missing','clarification',[{'role':'user','content':'I want to return my delivered desk lamp. I cannot remember my email. My first name is Maya and my last name is Ellis.'}],question='zip_only')
 add('email_dots','argument_copying',[{'role':'user','content':'Please check my order status. My email is dalia.moreno.eval@example.org. I do not have the order number.'}],{'name':'find_user_id_by_email','arguments':{'email':'dalia.moreno.eval@example.org'}})
 add('corrected_plus_email','argument_copying',[{'role':'user','content':'My email might be sam.kim+old@accounts.example.net. Please check my order.'},call('find_user_id_by_email',{'email':'sam.kim+old@accounts.example.net'}),obs('find_user_id_by_email',{'error':'User not found'}),{'role':'assistant','content':'That email did not match an account. Could you check it?'},{'role':'user','content':'That was the old address. Use sam.kim+orders@accounts.example.net instead.'}],{'name':'find_user_id_by_email','arguments':{'email':'sam.kim+orders@accounts.example.net'}})
 add('name_zip_leading_zero','argument_copying',[{'role':'user','content':'I need to return my order. I cannot remember my email. My first name is Ravi, my last name is Dutta, and my ZIP code is 02139.'}],{'name':'find_user_id_by_name_zip','arguments':{'first_name':'Ravi','last_name':'Dutta','zip':'02139'}})
 for i,uid in enumerate(('jules.park_eval_0042','customer_verified-007')):
  email=f'jules.park+{i}@example.org'
  prefix=[{'role':'user','content':f'Please check my order status. My email is {email}. I do not know the order number.'},call('find_user_id_by_email',{'email':email}),obs('find_user_id_by_email',uid)]
  add('after_auth_'+str(i),'post_auth_progression',prefix,{'name':'get_user_details','arguments':{'user_id':uid}})
 # These complete account/order observations come from real isolated Tau2 tools.
 for n,kind in ((808,'order'),(809,'cancel'),(810,'return')):
  tools,_,_,_,_,_=scenario(base,n);u=next(iter(tools.db.users.values()));o=next(iter(tools.db.orders.values()))
  o.status='delivered' if kind=='return' else 'pending'
  goal='check my order status' if kind=='order' else 'cancel my pending order because I no longer need it' if kind=='cancel' else 'return all items in my delivered order to the original payment method'
  prefix=[{'role':'user','content':f'I want to {goal}. My email is {u.email}. I do not know the order number.'},call('find_user_id_by_email',{'email':u.email}),obs('find_user_id_by_email',tools.find_user_id_by_email(u.email)),call('get_user_details',{'user_id':u.user_id}),obs('get_user_details',tools.get_user_details(u.user_id))]
  if kind=='order':add('order_id_from_account','post_auth_progression',prefix,{'name':'get_order_details','arguments':{'order_id':o.order_id}});continue
  prefix += [call('get_order_details',{'order_id':o.order_id}),obs('get_order_details',tools.get_order_details(o.order_id))]
  if kind=='cancel':
   prefix += [{'role':'assistant','content':f'Order {o.order_id} is pending. Shall I cancel it for the reason no longer needed?'},{'role':'user','content':f'Yes, cancel order {o.order_id}. I no longer need it.'}]
   expected={'name':'cancel_pending_order','arguments':{'order_id':o.order_id,'reason':'no longer needed'}}
  else:
   payment=o.payment_history[0].payment_method_id
   prefix += [{'role':'assistant','content':f'Shall I request the return of every item in order {o.order_id} using the original payment method {payment}?'},{'role':'user','content':f'Yes, return every item in order {o.order_id} using that original payment method.'}]
   expected={'name':'return_delivered_order_items','arguments':{'order_id':o.order_id,'item_ids':[x.item_id for x in o.items],'payment_method_id':payment}}
  add('confirmed_'+kind,'confirmed_execution',prefix,expected)
 add('corrected_name_and_zip','argument_copying',[{'role':'user','content':'Please help with my order. I cannot remember my email. My name is Nia Patel and ZIP is 07030.'},call('find_user_id_by_name_zip',{'first_name':'Nia','last_name':'Patel','zip':'07030'}),obs('find_user_id_by_name_zip',{'error':'User not found'}),{'role':'assistant','content':'No account matched. Could you check the name and ZIP?'},{'role':'user','content':'My surname was misspelled. First name Nia, last name Patil, ZIP code 07030. Please use those details.'}],{'name':'find_user_id_by_name_zip','arguments':{'first_name':'Nia','last_name':'Patil','zip':'07030'}})
 add('email_case_underscore','argument_copying',[{'role':'user','content':'Please find my order. My email is Elena_Ross+receipt@shop.example.com. I do not have the order number.'}],{'name':'find_user_id_by_email','arguments':{'email':'Elena_Ross+receipt@shop.example.com'}})
 assert len(cases)==12
 for case in cases:
  assert all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(case['messages'])),case['id']
 return cases

def score(case,text):
 try:calls=parse_calls(text);parse_error=None
 except ValueError as e:calls=[];parse_error=str(e)
 if case['expected_call'] is not None:
  expected=case['expected_call']
  if len(calls)==1 and expected['name']=='return_delivered_order_items' and calls[0].get('name')==expected['name']:
   from collections import Counter
   actual_args=dict(calls[0]['arguments']);expected_args=dict(expected['arguments'])
   items_ok=Counter(actual_args.pop('item_ids',[]))==Counter(expected_args.pop('item_ids',[]))
   passed=items_ok and actual_args==expected_args
  else:passed=calls==[expected]
  passed=passed and '<tool_response>' not in text
  return passed,{'parsed_calls':calls,'parse_error':parse_error,'rubric':'exact function and argument match'}
 q=text.lower();questions=re.findall(r'[^.!?\n]*\?',q)
 request=' '.join(questions) if questions else q
 if case['question']=='zip_only':
  passed=not calls and '<tool_call>' not in text and ('zip' in request or 'postal' in request) and not re.search(r'(?:first|last|full) name|email',request)
 else:passed=not calls and '<tool_call>' not in text and ('?' in text or 'provide' in q or 'share' in q) and ('email' in q or ('name' in q and ('zip' in q or 'postal' in q)))
 return passed,{'parsed_calls':calls,'parse_error':parse_error,'rubric':'rule-based clarification check; inspect wording in transcript'}

def main():
 os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
 cases=build_cases();system=json.loads((ROOT/'outputs/retail-eval/three-test-tasks-three-sft-stages/short-2k/task-38.json').read_text())['messages'][0]
 schemas=json.loads(re.search(r'<tools>(.*?)</tools>',system['content'],re.S)[1]);assert len(schemas)==17
 for case in cases:
  if case['expected_call']:
   c=case['expected_call'];validate(c['arguments'],next(t['function']['parameters'] for t in schemas if t['function']['name']==c['name']))
 atomic(OUT/'cases.json',cases)
 atomic(OUT/'config.json',{'models':MODELS,'seeds':[42,43,44],'temperature':.7,'top_p':.95,'max_new_tokens':256,'tool_count':17,'shared_system_prompt':system['content'],'rank_by':'mean of four skill-category pass rates, then argument-copying pass rate','note':'Controlled next-action diagnostics, not full Tau2 task rewards. Same 12 cases independently supplied to every model; no conditional model-history propagation.'})
 rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
 def save(status):
  atomic(OUT/'results.json',rows);atomic(OUT/'progress.json',{'status':status,'completed':len(rows),'target':108})
  summaries=[];transcripts=[]
  for label,_ in MODELS:
   selected=[r for r in rows if r['model']==label];categories={}
   for cat in sorted({c['category'] for c in cases}):
    group=[r for r in selected if r['category']==cat];categories[cat]={'passed':sum(r['passed'] for r in group),'total':len(group),'rate':sum(r['passed'] for r in group)/len(group) if group else None}
   complete=len(selected)==36;macro=sum(v['rate'] for v in categories.values())/4 if complete else None
   summaries.append({'model':label,'passed':sum(r['passed'] for r in selected),'total':len(selected),'complete':complete,'categories':categories,'macro_pass_rate':macro})
   text='=== '+label+' ===\n\n'+'\n\n'.join(f"CASE: {r['case_id']} | SEED: {r['seed']} | PASS: {r['passed']}\nEXPECTED: {json.dumps(r['expected_call']) if r['expected_call'] else r['question']}\n\n"+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'][1:]) for r in selected)
   (OUT/(label+'.txt')).write_text(text);transcripts.append(text)
  atomic(OUT/'summary.json',summaries);(OUT/'conversations.txt').write_text('\n\n'.join(transcripts))
  if len(rows)==108:
   ranked=sorted(summaries,key=lambda r:(r['macro_pass_rate'],r['categories']['argument_copying']['rate'],r['passed']),reverse=True)
   atomic(OUT/'recommendation.json',{'selected_model':ranked[0]['model'] if ranked[0]['passed'] else None,'ranking':ranked,'scope':'Best observed candidate on this diagnostic set only; not a claim of reliable full-task performance.'})
 for label,adapter in MODELS:
  if sum(r['model']==label for r in rows)==36:continue
  save('loading '+label);print('LOADING',label,flush=True)
  policy=LocalPolicy(RetailRLConfig(initial_adapter=adapter or '',temperature=.7,top_p=.95),adapter=adapter is not None)
  for case in cases:
   messages=[system]+case['messages'];ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False);assert len(ids)+256<=8192
   for seed in (42,43,44):
    if any(r['model']==label and r['case_id']==case['id'] and r['seed']==seed for r in rows):continue
    result=policy.generate({'prompt_ids':ids},256,seed);passed,details=score(case,result['text'])
    row={'model':label,'checkpoint':adapter,'case_id':case['id'],'category':case['category'],'seed':seed,'passed':passed,'expected_call':case['expected_call'],'question':case['question'],'output':result['text'],'messages':messages+[{'role':'assistant','content':result['text']}],'ended_with_eos':result['ended_with_eos'],'tokens':len(result['completion_ids']),'prompt_tokens':len(ids),'native_eos3_inside_output':3 in result['completion_ids'],**details}
    rows.append(row);save('testing '+label);print('RESULT',label,case['id'],seed,passed,flush=True)
  del policy;gc.collect();torch.cuda.empty_cache()
 save('complete');print('DONE',OUT,flush=True)

if __name__=='__main__':main()
