"""Diverse exact-copy curriculum, grounded using real isolated Tau2 retail tools."""
import concurrent.futures,json,re,threading,os
from pathlib import Path
from openai import OpenAI
from jsonschema import validate
from scripts.generate_long_retail_sft import prepare
from tau2_grpo.prepare_function_calling import digest
from tau2_grpo.sft_data import tokenize_assistant_only
OUT=Path('outputs/retail-data/copy-repair-500')
PROMPT='''Rewrite short retail dialogue slots naturally. Return JSON {"slots":{"key":"text",...}}, string values only. Preserve every required substring EXACTLY, including dots, hyphens, underscores, plus signs, case and leading zeros. An email, ZIP, name, user ID or order ID is opaque data: never invent digits, normalize punctuation, change a domain, or turn a user ID into an email. Do not introduce facts into earlier messages. Ask ONLY for missing information specified by each question. call_note slots state intent in one short sentence, not success. No extra questions and no tool markup. The driver executes and serializes the correct calls. Keep each slot to one or two short sentences.'''

def email_for(n,first,last):
 k=(n//10)%10;tag=chr(97+n//26)+chr(97+n%26);f=first.lower()+tag;l=last.lower().replace(' ','').replace("'",'')
 return [f'{f}@example.org',f'{f}.{l}@example.com',f'{f}_{l}@example.net',f'{f}+receipts@mail.example.org',f'{f}-{l}@shop.example.com',f'{first}.{last}.{tag.upper()}@example.net',f'{f}.{l}.007@example.org',f'{f}09_{l}@mail.example.net',f'{f}+order-03@example.com',f'{f}.{l}@example.org'][k]

def scenario(base,n):
 tools,schemas,system,_,_,_,_=prepare(base,70000+n)
 user=next(iter(tools.db.users.values()));order=next(iter(tools.db.orders.values()))
 first=['Lena','Amir','Celine','Nico','Zoya','Evan','Anika','Hugo','Iris','Lucia'][(n//10+n%10+n//100)%10]
 last=['Avery','O\'Neil','Al-Salem','de Vries','Muñoz','Bennett','Dubois','Iversen','Petrov','Silva'][(n//10+n%10+n//100)%10]
 user.name.first_name=first;user.name.last_name=last;user.email=email_for(n,first,last)
 user.address.zip=['00501','00926','04101','07030','10001','30303','60614','85004','94107','02108'][(n//10)%10]
 # Include unseen leading-zero values while avoiding exact earlier evaluation ZIPs.
 if n//100:user.address.zip=f'{(n*37+1001)%99999:05d}'
 order.address=user.address.model_copy(deep=True)
 oldid=user.user_id
 uid=[f'{first.lower()}_{n:03d}',f'{first.lower()}.{last.lower().replace(" ","_")}.{chr(97+n//26)}{chr(97+n%26)}',f'account-{n:04d}-verified',f'{first.lower()}_member_{chr(97+n//26)}{chr(97+n%26)}'][n%4]
 user.user_id=uid;order.user_id=uid;tools.db.users={uid:user}
 oldorder=order.order_id;oid=f'#W{8100000+n*13}';order.order_id=oid;user.orders=[oid];tools.db.orders={oid:order}
 slots={};steps=[]
 def text(role,key,value,required=()):slots[key]={'text':value,'required':list(required)};steps.append(('text',role,key))
 def call(name,args,last=False):
  key='call_note_'+str(sum(s[0]=='call' for s in steps));slots[key]={'text':{'find_user_id_by_email':"I'll look up your account using that exact email.",'find_user_id_by_name_zip':"I'll look up your account using your name and ZIP code.",'get_user_details':"I'll retrieve your account details using the returned user ID.",'get_order_details':"I'll retrieve the order from your account's order list."}[name],'required':[]};steps.append(('call',name,args,key,last))
 kind=n%10;lookup={'name':'find_user_id_by_email','arguments':{'email':user.email}}
 if kind<4:
  profile='email_exact_copy';text('user','opening','I need help with an order. I do not know the order number.')
  text('assistant','ask_email','What is your email address?',['email'])
  if kind==3:
   profile='corrected_email';bad=user.email.replace('@','-typo@');text('user','typo',f'I think it is {bad}.',[bad]);call('find_user_id_by_email',{'email':bad});text('assistant','recover','No account matched that email. Could you check the spelling?',['email']);text('user','correction',f'The correct email is {user.email}. Please use that exact address.',[user.email])
  else:text('user','identity',f'My email is {user.email}.',[user.email])
  call(lookup['name'],lookup['arguments'],True)
 elif kind<6:
  profile='name_zip_exact_copy';text('user','opening',f'I need help with an order. I cannot remember my email. My first name is {first}, my last name is {last}, and my ZIP code is {user.address.zip}.',[first,last,user.address.zip]);call('find_user_id_by_name_zip',{'first_name':first,'last_name':last,'zip':user.address.zip},True)
 elif kind<8:
  profile='only_zip_missing';text('user','opening',f'I need help with an order. I cannot remember my email. My first name is {first} and my last name is {last}.',[first,last]);text('assistant','ask_zip','What is your ZIP code?',['ZIP']);text('user','zip',f'My ZIP code is {user.address.zip}.',[user.address.zip]);call('find_user_id_by_name_zip',{'first_name':first,'last_name':last,'zip':user.address.zip},True)
 else:
  profile='user_id_after_auth' if kind==8 else 'order_id_from_observation'
  text('user','opening',f'Please help me check my order. My email is {user.email}. I do not know the order number.',[user.email]);call('find_user_id_by_email',{'email':user.email});call('get_user_details',{'user_id':uid},kind==8)
  if kind==9:call('get_order_details',{'order_id':oid},True)
 return tools,schemas,system,slots,steps,profile

def main():
 from tau2.domains.retail.data_model import get_db
 from transformers import AutoTokenizer
 OUT.mkdir(parents=True,exist_ok=True)
 for d in ('records','attempts'):(OUT/d).mkdir(exist_ok=True)
 base=get_db();tokenizer=AutoTokenizer.from_pretrained('outputs/hf-export/tiny-aya-global-function-calling-sft-525')
 client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=180,max_retries=2)
 (OUT/'generator-prompt.txt').write_text(PROMPT)
 (OUT/'split-manifest.json').write_text(json.dumps([{'index':n,'split':'validation' if (n//10)%10==9 else 'train','family':f'copy-repair-{n:04d}'} for n in range(500)],indent=2))
 def export():
  rows=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))]
  for name,split in [('sft','train'),('validation','validation'),('all',None)]:
   tmp=OUT/(name+'.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows if split is None or r['split']==split));tmp.replace(OUT/(name+'.jsonl'))
  status={'samples':len(rows),'target':500,'complete':len(rows)==500,'train':sum(r['split']=='train' for r in rows),'validation':sum(r['split']=='validation' for r in rows),'max_context_tokens':max((r['context_tokens'] for r in rows),default=0),'profiles':sorted(set(r['profile'] for r in rows))}
  (OUT/'progress.json').write_text(json.dumps(status,indent=2))
  if len(rows)==500:(OUT/'report.json').write_text(json.dumps(status,indent=2))
  return status
 for n in range(500):
  dest=OUT/'records'/f'{n:04d}.json'
  if dest.exists():continue
  for attempt in range(25):
   try:
    tools,schemas,system,slots,steps,profile=scenario(base,n)
    request=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps({'slots':slots,'profile':profile})}]
    response=client.chat.completions.create(model='qwen3.5-4b',messages=request,temperature=.5,max_tokens=1600,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    (OUT/'attempts'/f'{n:04d}-{attempt}.json').write_text(json.dumps({'messages':request,'response':response.model_dump()},indent=2))
    assert response.choices[0].finish_reason!='length','truncated output'
    words=json.loads(response.choices[0].message.content)['slots'];assert set(words)==set(slots)
    words={k:(v['text'] if isinstance(v,dict) and 'text' in v else v) for k,v in words.items()}
    protected={str(v) for step in steps if step[0]=='call' for v in step[2].values()}
    for key,spec in slots.items():
     value=words[key];assert isinstance(value,str) and 0<len(value)<650
     assert all(s in value for s in spec['required']),f'changed required value: {key}'
     for fact in protected:
      if fact in value:assert fact in spec['text'],f'early/invented fact in {key}'
     for token in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d{5,}\b',value):assert token in spec['text'],f'invented identifier {key}'
     assert '<tool_' not in value
     if key=='ask_zip':assert 'name' not in value.lower() and 'email' not in value.lower(),'asking known fields'
     if key.startswith('call_note_'):assert '?' not in value and not re.search(r'(?:found|located|authenticated|successfully)',value,re.I),'premature result'
    messages=[{'role':'system','content':system}];audit=[];visible='';authenticated=False
    for step in steps:
     if step[0]=='text':
      _,role,key=step;messages.append({'role':role,'content':words[key]})
      if role=='user':visible+=' '+words[key]
     else:
      _,name,args,key,last=step;validate(args,next(s['function']['parameters'] for s in schemas if s['function']['name']==name))
      assert all(str(v) in visible for v in args.values()),'tool argument is not observed'
      if name in ('get_user_details','get_order_details'):assert authenticated
      messages.append({'role':'assistant','content':words[key]+'\n<tool_call>'+json.dumps({'name':name,'arguments':args},separators=(',',':'),ensure_ascii=False)+'</tool_call>'})
      try:
       result=getattr(tools,name)(**args);body=result.model_dump(mode='json') if hasattr(result,'model_dump') else result;error=False
       if name.startswith('find_user'):authenticated=True
      except ValueError as exc:
       assert profile=='corrected_email' and not authenticated and name=='find_user_id_by_email';body={'error':str(exc)};error=True
      audit.append({'name':name,'arguments':args,'result':body,'error':error})
      if not last:messages.append({'role':'user','content':'<tool_response>'+json.dumps({'name':name,'content':body},separators=(',',':'),ensure_ascii=False)+'</tool_response>'});visible+=' '+json.dumps(body,ensure_ascii=False)
    assert not audit[-1]['error'];indices=[i for i,m in enumerate(messages) if m['role']=='assistant'];encoded,_=tokenize_assistant_only(tokenizer,messages,8192,indices)
    family=f'copy-repair-{n:04d}';record={'id':family,'task_id':family,'domain':'retail','group_id':digest({'domain':'retail','task_id':family}),'split':'validation' if (n//10)%10==9 else 'train','profile':profile,'messages':messages,'loss_message_indices':indices,'tool_execution_audit':audit,'guided_synthetic':True,'context_tokens':len(encoded['input_ids']),'checks':{'exact_argument_grounding':True,'executed_tools':True,'assistant_masking':True,'within_8k':True}}
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(dest);dest.with_suffix('.txt').write_text('\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in messages[1:]));status=export();print('ACCEPTED',n,profile,status['samples'],flush=True);break
   except Exception as exc:
    (OUT/'attempts'/f'{n:04d}-{attempt}-error.txt').write_text(repr(exc));print('RETRY',n,attempt,repr(exc),flush=True)
  else:raise RuntimeError('Generation stopped after repeated failures; inspect rejected attempts before resume')
 print('DONE',json.dumps(export()),flush=True)

if __name__=='__main__':main()
