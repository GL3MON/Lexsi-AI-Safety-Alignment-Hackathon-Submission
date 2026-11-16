#!/usr/bin/env python3
"""Resumable, grounded short clarification curriculum across local Qwen servers."""
import argparse,concurrent.futures,json,re,threading,hashlib
from pathlib import Path
from openai import OpenAI
from jsonschema import validate
from tau2_grpo.prepare_function_calling import digest
from tau2_grpo.sft_data import tokenize_assistant_only
from scripts.generate_long_retail_sft import prepare

PROMPT='''Rewrite the provided short retail dialogue slots naturally. Return JSON {"slots":{"slot_name":"text",...}}. Each slot value must be a string. Preserve meaning and ALL required substrings exactly. Do not add facts, identity values, success claims, questions, or additional requirements. Questions ask ONLY for the fields in the original slot. Do not ask again for already supplied information. The call_note slot is a brief statement of intended action, never a claim that the account is already found or authenticated. The driver appends the real tool-call block to that SAME assistant message. Never generate tool markup yourself. Keep each slot concise, normally one sentence.'''

def specification(n):
 pos=n%100
 if pos<60:variant='ask_then_act';index=(n//100)*60+pos
 elif pos<87:variant='ready_to_act';index=(n//100)*27+pos-60
 else:variant='partial_answer';index=(n//100)*13+pos-87
 return variant,('validation' if index%10==9 else 'train')

def run(args):
 from tau2.domains.retail.data_model import get_db
 from transformers import AutoTokenizer
 base=get_db();tok=AutoTokenizer.from_pretrained(args.tokenizer);lock=threading.Lock()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 for name in ('records','attempts'):(out/name).mkdir(exist_ok=True)
 config=vars(args)|{'teacher_prompt':PROMPT,'guided_synthetic':True,'scope':'identity clarifications, ready-to-call contrasts, partial identity answers','allocation':{'ask_then_act':1200,'ready_to_act':540,'partial_answer':260},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 cp=out/'config.json'
 if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Resume config mismatch')
 cp.write_text(json.dumps(config,indent=2));(out/'generator-prompt.txt').write_text(PROMPT)
 specs=[{'index':n,'variant':specification(n)[0],'split':specification(n)[1],'endpoint':args.endpoints[n%len(args.endpoints)]} for n in range(args.count)]
 (out/'split-manifest.json').write_text(json.dumps(specs,indent=2))
 def export():
  records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
  for name,split in [('sft','train'),('validation','validation'),('all',None)]:
   tmp=out/(name+'.jsonl.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in records if split is None or r['split']==split));tmp.replace(out/(name+'.jsonl'))
  status={'target':args.count,'accepted':len(records),'complete':len(records)==args.count,'train':sum(r['split']=='train' for r in records),'validation':sum(r['split']=='validation' for r in records),'variants':{v:sum(r['variant']==v for r in records) for v in ('ask_then_act','ready_to_act','partial_answer')},'endpoints':{e:sum(r['endpoint']==e for r in records) for e in args.endpoints},'max_context_tokens':max((r['context_tokens'] for r in records),default=0)}
  (out/'progress.json').write_text(json.dumps(status,indent=2))
  return status
 def worker(w):
  client=OpenAI(base_url=args.endpoints[w],api_key='local',timeout=180,max_retries=2)
  for n in range(w,args.count,len(args.endpoints)):
   dest=out/'records'/f'{n:05d}.json'
   if dest.exists():continue
   variant,split=specification(n)
   tools,schemas,system,_,_,_,action=prepare(base,10000+n*10+(5 if n%2 else 0))
   user=next(iter(tools.db.users.values()));first=user.name.first_name;last=user.name.last_name;zipcode=user.address.zip
   use_email=(n//2)%2==0 and variant!='partial_answer'
   name='find_user_id_by_email' if use_email else 'find_user_id_by_name_zip'
   arguments={'email':user.email} if use_email else {'first_name':first,'last_name':last,'zip':zipcode}
   goal='cancel my order because I no longer need it' if action=='cancel' else 'return all items in my delivered order to the original payment method'
   slots={};sequence=[]
   def add(role,key,text,required=()):slots[key]={'text':text,'required':list(required)};sequence.append((role,key))
   if variant=='ready_to_act':
    identity=f'My email is {user.email}.' if use_email else f'My first name is {first}, my last name is {last}, and my ZIP code is {zipcode}. I cannot remember my email.'
    add('user','opening',f'I want to {goal}. I do not know the order number. '+identity,arguments.values())
   else:
    add('user','opening',f'I want to {goal}. I do not know the order number.'+(' I cannot remember my email.' if not use_email else ''))
    question='Could you provide your email address?' if use_email else 'Could you provide your first name, last name, and ZIP code?'
    add('assistant','ask',question,['email'] if use_email else ['first name','last name','ZIP'])
    if variant=='partial_answer':
     add('user','partial',f'My full name is {first} {last}.',[first,last]);add('assistant','ask_zip','What is your ZIP code?',['ZIP']);add('user','identity',f'My ZIP code is {zipcode}.',[zipcode])
    else:
     identity=f'My email is {user.email}.' if use_email else f'My first name is {first}, my last name is {last}, and my ZIP code is {zipcode}.'
     add('user','identity',identity,arguments.values())
   slots['call_note']={'text':"I'll look up your account using that email." if use_email else "I'll look up your account using your name and ZIP code.",'required':[]}
   for attempt in range(args.retries):
    try:
     request=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps({'variant':variant,'slots':slots})}]
     response=client.chat.completions.create(model=args.model,messages=request,temperature=.6,seed=42+n+attempt*args.count,max_tokens=1100,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
     (out/'attempts'/f'{n:05d}-{attempt}.json').write_text(json.dumps({'messages':request,'response':response.model_dump()},indent=2))
     assert response.choices[0].finish_reason!='length','truncated output'
     words=json.loads(response.choices[0].message.content)['slots'];assert set(words)==set(slots),'missing/extra slots'
     for key,spec in slots.items():
      value=words[key];assert isinstance(value,str) and 0<len(value)<650,'invalid slot text'
      assert all(x in value for x in spec['required']),'required facts omitted'
      assert '<tool_' not in value,'model-generated markup'
      for token in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d{5,}\b',value):assert token in spec['text'],'invented/early identifier'
      assert not re.search(r'(?:found|located|authenticated|successfully|has been cancelled|has been returned)',value,re.I),'unsupported success claim'
      if key=='ask_zip':assert 'name' not in value.lower() and 'email' not in value.lower(),'asking for known fields'
     assert '?' not in words['call_note'],'question instead of intent'
     messages=[{'role':'system','content':system}]+[{'role':role,'content':words[key]} for role,key in sequence]
     visible=' '.join(m['content'] for m in messages if m['role']=='user')
     assert all(str(v) in visible for v in arguments.values()),'unseen call argument'
     # Execute the lookup against this isolated account as a grounding check.
     validate(arguments,next(s['function']['parameters'] for s in schemas if s['function']['name']==name))
     assert getattr(tools,name)(**arguments)==user.user_id
     messages.append({'role':'assistant','content':words['call_note']+'\n<tool_call>'+json.dumps({'name':name,'arguments':arguments},separators=(',',':'))+'</tool_call>'})
     indices=[i for i,m in enumerate(messages) if m['role']=='assistant']
     with lock:enc,_=tokenize_assistant_only(tok,messages,8192,indices)
     family=f'short-clarification-synthetic-{n:05d}'
     record={'id':family,'task_id':family,'domain':'retail','group_id':digest({'domain':'retail','task_id':family}),'scenario_family':family,'split':split,'variant':variant,'messages':messages,'loss_message_indices':indices,'guided_synthetic':True,'context_tokens':len(enc['input_ids']),'supervised_tokens':sum(x!=-100 for x in enc['labels']),'endpoint':args.endpoints[w],'lookup_verified':True}
     tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(dest)
     dest.with_suffix('.txt').write_text('\n\n'.join(('ASSISTANT TOOL CALL' if '<tool_call>' in m['content'] else m['role'].upper())+': '+m['content'] for m in messages[1:]))
     with lock:
      status=export();print('ACCEPTED',n,variant,split,'total',status['accepted'],flush=True)
     break
    except Exception as exc:
     (out/'attempts'/f'{n:05d}-{attempt}-error.txt').write_text(repr(exc));print('RETRY',n,attempt,repr(exc),flush=True)
   else:raise RuntimeError(f'Sample {n} failed; resume after inspecting rejected attempts')
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.endpoints)) as pool:
  futures=[pool.submit(worker,w) for w in range(len(args.endpoints))]
  for future in futures:future.result()
 status=export();assert status['complete'];print('DONE',json.dumps(status),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=2000);p.add_argument('--output',default='outputs/retail-data/short-clarification-2000');p.add_argument('--endpoints',nargs='+',default=['http://127.0.0.1:8005/v1','http://127.0.0.1:8004/v1']);p.add_argument('--model',default='qwen3.5-4b');p.add_argument('--tokenizer',default='outputs/hf-export/tiny-aya-global-function-calling-sft-525');p.add_argument('--retries',type=int,default=20);run(p.parse_args())
