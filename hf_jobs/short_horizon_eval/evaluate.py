import json,re,time,hashlib,os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from peft import PeftModel
from tau2.domains.retail.data_model import get_db,RetailDB
from tau2.domains.retail.tools import RetailTools
work=Path(os.environ['SHORT_WORK']);project=work/'project';adapter=project/'initial-adapter';out=Path(os.environ['SHORT_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest();assert sha=='2c72a97c0abd990dd87be650ace9674051b4e30dfff8269f1e0b8a4980b0a7f2'
tokenizer=AutoTokenizer.from_pretrained(adapter);system=(project/'templates/retail_apigen_sft_system.txt').read_text();assert tokenizer.chat_template==(project/'templates/retail_sdg_v2.jinja').read_text()
model=AutoModelForCausalLM.from_pretrained('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa');model=PeftModel.from_pretrained(model,adapter,is_trainable=False);model.eval()
generation=GenerationConfig(do_sample=False,max_new_tokens=128,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,use_cache=True)
base=get_db();first,last,email,zip_code,uid,oid='Zehra','Marin','zehra.marin@example.org','02134','zehra_marin_short_7416','#W4827159'
def world():
 source=next(o for o in sorted(base.orders.values(),key=lambda x:x.order_id) if o.status=='pending' and o.items and o.payment_history)
 customer=base.users[source.user_id].model_copy(deep=True);order=source.model_copy(deep=True)
 customer.user_id=uid;customer.name.first_name=first;customer.name.last_name=last;customer.email=email;customer.address.zip=zip_code;customer.orders=[oid]
 order.order_id=oid;order.user_id=uid;order.address=customer.address.model_copy(deep=True)
 return RetailTools(RetailDB(users={uid:customer},orders={oid:order},products={i.product_id:base.products[i.product_id].model_copy(deep=True) for i in order.items}))
def call(name,args):return dict(name=name,arguments=args)
def render(text):
 def f(m):
  try:return '<tool_call>\n'+json.dumps(json.loads(m.group(1)),indent=2)+'\n</tool_call>'
  except ValueError:return m.group(0)
 return re.sub(r'<tool_call>(.*?)</tool_call>',f,text,flags=re.S)
results=[]
def save():
 (out/'results.json').write_text(json.dumps(results,indent=2))
 (out/'conversations.txt').write_text('\n\n'.join('=== '+r['name']+' | PASS='+str(r['passed'])+' ===\n'+ '\n\n'.join(m['role'].upper()+':\n'+render(m['content']) for m in r['messages'] if m['role']!='system')+'\n\nCHECKS: '+json.dumps(r['checks']) for r in results))
for name,opening in [('email_lookup',f'Please find my customer account. My email is {email}.'),('missing_zip',f'I cannot remember my email. My first name is {first} and my last name is {last}. Please find my account.'),('order_status',f'Please check the current status of my order. I do not know its number. My email is {email}.')]:
 tools=world();messages=[dict(role='system',content=system),dict(role='user',content=opening)];checks=[];prompts=[];started=time.monotonic();passed=True
 if name=='email_lookup':steps=[call('find_user_id_by_email',{'email':email})]
 elif name=='missing_zip':steps=['ask_zip',call('find_user_id_by_name_zip',dict(first_name=first,last_name=last,zip=zip_code))]
 else:steps=[call('find_user_id_by_email',{'email':email}),call('get_user_details',{'user_id':uid}),call('get_order_details',{'order_id':oid}),'report_status']
 for turn,expected in enumerate(steps,1):
  inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0');assert inputs['input_ids'].shape[1]+128<=8192
  with torch.inference_mode():tokens=model.generate(**inputs,generation_config=generation,logits_to_keep=1)[0,inputs['input_ids'].shape[1]:]
  text=tokenizer.decode(tokens,skip_special_tokens=True);messages.append(dict(role='assistant',content=text));prompts.append(int(inputs['input_ids'].shape[1]));ended=bool(len(tokens) and tokens[-1].item()==tokenizer.eos_token_id)
  blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
  try:parsed=[json.loads(b) for b in blocks]
  except ValueError:parsed=None
  ok=ended and text.count('<tool_call>')==len(blocks) and text.count('</tool_call>')==len(blocks)
  if expected=='ask_zip':
   request=' '.join(re.findall(r'(?:could you|can you|please|what (?:is|are)|do you)[^?.!]*(?:[?.!]|$)',text.lower()))
   ok=ok and parsed==[] and bool(re.search(r'\bzip\b|postal',request)) and not bool(re.search(r'first name|last name|surname|email|other information|additional',request)) and not bool(re.search(r'i (?:have )?(?:found|located|verified)',text,re.I))
  elif expected=='report_status':ok=ok and parsed==[] and oid in text and bool(re.search(r'\bpending\b|not.*processed|awaiting.*process',text,re.I))
  else:
   ok=ok and parsed==[expected]
   if ok:
    try:
     observed=getattr(tools,expected['name'])(**expected['arguments']);value=observed.model_dump(mode='json') if hasattr(observed,'model_dump') else observed
     messages.append(dict(role='tool',name=expected['name'],content=json.dumps(dict(name=expected['name'],content=value),separators=(',',':'))))
    except Exception as error:ok=False;checks.append(dict(execution_error=str(error)))
  checks.append(dict(turn=turn,expected=expected,passed=bool(ok)));print(json.dumps(dict(case=name,turn=turn,passed=bool(ok),text=text)),flush=True)
  if not ok:passed=False;break
  if expected=='ask_zip':messages.append(dict(role='user',content=f'My ZIP code is {zip_code}.'))
 results.append(dict(name=name,passed=bool(passed),messages=messages,checks=checks,actor_turns=len(prompts),max_prompt_tokens=max(prompts),seconds=time.monotonic()-started));save()
summary=dict(checkpoint='run-2190c37f90/checkpoint-25',adapter_sha256=sha,passed=sum(r['passed'] for r in results),total=3,backend='transformers',dtype='bf16',temperature=0,max_new_tokens=128,gpu=torch.cuda.get_device_name(),note='Controlled short-horizon diagnostics with real RetailTools and fixed customer replies. Same SFT system/template. These are not native Tau2 tasks or rewards; passing does not establish long-horizon ability.')
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
