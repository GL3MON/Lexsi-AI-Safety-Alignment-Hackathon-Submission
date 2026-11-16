"""Verified long-horizon retail curriculum with masked faults and real tools.

Qwen supplies wording, while a deterministic expert plan executes isolated Tau2
DB operations. These are synthetic expert demonstrations, not benchmark scores.
"""
import argparse,copy,hashlib,json,re
from decimal import Decimal
from pathlib import Path
from openai import OpenAI
from jsonschema import validate
from transformers import AutoTokenizer
from tau2_grpo.prepare_function_calling import parse_calls,digest
from tau2_grpo.sft_data import tokenize_assistant_only
from tau2_grpo.rl_rollout import atomic_json

NOTES={
 'ask_email':('Could you provide your email address so I can find your account?',['email']),
 'ask_zip':('I have your name. What is your ZIP code?',['ZIP']),
 'email_retry':('That email did not match an account. Please check it, or provide your name and ZIP code.',['email','ZIP']),
 'repair_read':('The request did not execute because its JSON was incomplete. I will resend a complete JSON request using the identifiers already returned by the tools.',['JSON']),
 'repair_write':('The request did not execute because its JSON was incomplete. Your confirmation still applies; I will resend the corrected request.',['JSON','confirmation']),
 'repair_argument':('The tool rejected the argument name. I will use order_id and copy the identifier from the account result.',['order_id']),
 'delivered':('The tool rejected cancellation because this order is delivered. No cancellation occurred; the supported option is a return.',['delivered','return']),
 'repair_payment':('The tool rejected that payment identifier. I will copy the original payment method from this order and retry the confirmed return.',['original payment']),
 'late_repair':('The tool rejected the expression. I will use the actual prices, rather than identifiers, in the calculation.',['prices','identifiers']),
 'read_account':('I will retrieve the account details and order list using the verified user identifier.',['account']),
 'read_order':('I will inspect this order and use its actual status, item identifiers, and original payment method.',['status']),
 'catalog':('I will inspect the available variants and compare size, material, availability, and price before choosing an exchange.',['size','material','price']),
}
WORDING_PROMPT='''Rewrite the supplied retail assistant sentences naturally. Return JSON {"notes":{"key":"sentence"}} with exactly the provided keys. Preserve meaning, required substrings, and the distinction between failed execution, planned action, and completed action. Do not add any customer facts, identifiers, tools, monetary amounts, claims of success, extra questions, or tool markup. Recovery sentences must explain the actual error and the specific grounded next step. Maximum 60 words per entry. These sentences will accompany real tool calls; do not generate calls.'''


def wording(client,model,out,n):
 for attempt in range(5):
  request=[{'role':'system','content':WORDING_PROMPT},{'role':'user','content':json.dumps({k:{'text':v,'required':required} for k,(v,required) in NOTES.items()})}]
  response=client.chat.completions.create(model=model,messages=request,temperature=.4,max_tokens=1800,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
  atomic_json(out/'attempts'/f'{n:03d}-wording-{attempt}.json',{'request':request,'response':response.model_dump()})
  try:
   if response.choices[0].finish_reason=='length':raise ValueError('truncated wording')
   notes=json.loads(response.choices[0].message.content)['notes']
   if set(notes)!=set(NOTES):raise ValueError('wrong note keys')
   for k,text in notes.items():
    if not isinstance(text,str) or not 4<=len(text.split())<=60:raise ValueError('invalid note length')
    if not all(x in text for x in NOTES[k][1]):raise ValueError('required meaning marker missing: '+k)
    if re.search(r'\d|@|<tool|successfully|has been completed',text,re.I):raise ValueError('invented facts or premature success')
    if k in ('repair_read','repair_write') and not re.search(r'not execute|failed|didn.t execute|not processed|not run|not executed',text,re.I):raise ValueError('missing failed-execution statement')
   return notes
  except (KeyError,ValueError,TypeError) as e:
   print('WORDING RETRY',n,attempt,str(e),flush=True)
 raise RuntimeError('Qwen wording failed validation')


def make_world(seed,n):
 from tau2.domains.retail.data_model import RetailDB,OrderPayment
 from tau2.domains.retail.tools import RetailTools
 candidates=[];seen=set()
 for o in sorted(seed.orders.values(),key=lambda x:x.order_id):
  if o.items and o.payment_history and o.items[0].name not in seen:
   candidates.append(o);seen.add(o.items[0].name)
 boot=next(o for o in seed.orders.values() if o.items and o.items[0].name=='Hiking Boots')
 others=[o for o in candidates if o.items[0].name!='Hiking Boots'];chosen=[others[(n*5+i)%len(others)] for i in range(5)]+[boot]
 user=seed.users[chosen[0].user_id].model_copy(deep=True)
 first=['Dalia','Kian','Mireya','Arun','Nadia','Emil','Soraya','Luca','Priya','Tomas','Ines','Zaid'][n%12]
 last=['Moreno','Chen','Vega','Menon','Farouk','Lind','Navarro','Costa','Shah','Novak','Duarte','Haddad'][n%12]
 user.user_id=f'horizon_customer_{n:04d}';user.name.first_name=first;user.name.last_name=last
 user.email=f'{first.lower()}.{last.lower()}+receipts-{n:02d}@example.org';user.address.zip=f'{1050+n:05d}'
 method=next(iter(user.payment_methods.values())).model_copy(deep=True);method.id=f'{method.source}_{9700000+n}';user.payment_methods={method.id:method}
 orders={};products={};objects=[];cheaper=n%3!=2
 for i,source in enumerate(chosen):
  order=source.model_copy(deep=True);item=order.items[0].model_copy(deep=True);product=seed.products[item.product_id].model_copy(deep=True)
  oldvariant=product.variants[item.item_id].model_copy(deep=True)
  product.product_id=str(9100000000+n*20+i);item.product_id=product.product_id
  item.item_id=str(9200000000+n*100+i*10);oldvariant.item_id=item.item_id
  item.price=round(110.25+n*3.1+i*23.4,2);oldvariant.price=item.price;oldvariant.available=True
  product.variants={item.item_id:oldvariant}
  if i==5:
   item.options.update(size=str(8+n%4),material='leather');oldvariant.options=copy.deepcopy(item.options)
   good=oldvariant.model_copy(deep=True);good.item_id=str(int(item.item_id)+1);good.price=round(item.price-17.25 if cheaper else item.price+12.5,2);good.available=True
   if 'waterproof' in good.options:good.options['waterproof']='no' if good.options['waterproof']=='yes' else 'yes'
   wrong=oldvariant.model_copy(deep=True);wrong.item_id=str(int(item.item_id)+2);wrong.options['size']=str(int(item.options['size'])+1);wrong.price=round(item.price-25,2)
   product.variants.update({good.item_id:good,wrong.item_id:wrong})
  products[product.product_id]=product
  order.order_id=f'#W{7600000+n*10+i}';order.user_id=user.user_id;order.address=user.address.model_copy(deep=True);order.items=[item]
  order.status='pending' if i in (0,1,4) else 'delivered';order.fulfillments=[]
  for key in ('cancel_reason','return_items','return_payment_method_id','exchange_items','exchange_new_items','exchange_payment_method_id','exchange_price_difference'):setattr(order,key,None)
  order.payment_history=[OrderPayment(transaction_type='payment',amount=item.price,payment_method_id=method.id)]
  orders[order.order_id]=order;objects.append(order)
 user.orders=list(orders)
 return RetailTools(RetailDB(users={user.user_id:user},orders=orders,products=products)),user,objects,method.id,cheaper


class Trace:
 def __init__(self,tools,system,notes):
  self.tools=tools;self.messages=[{'role':'system','content':system}];self.targets=[];self.audit=[];self.faults=[];self.notes=notes;self.authorized=set();self.customer_facts=[];self.plan=[]
  self.schemas={t.name:t.openai_schema['function']['parameters'] for t in tools.get_tools().values()}
 def add(self,role,text,target=True):
  assert role!=self.messages[-1]['role'],'roles must alternate'
  index=len(self.messages);self.messages.append({'role':role,'content':text})
  if role=='assistant' and target:self.targets.append(index)
  if role=='user' and not text.startswith('<tool_response>'):self.customer_facts.append(text)
  return index
 def dialogue(self,assistant,user):self.add('assistant',assistant);self.add('user',user)
 def confirm(self,name,args,text):
  self.dialogue(text,'Yes, please proceed exactly as you described.')
  self.authorized.add((name,json.dumps(args,sort_keys=True)))
 def call(self,name,args,note,*,fault=None,malformed=False):
  write=name.startswith(('cancel_','modify_','return_','exchange_'))
  if write and not fault:assert (name,json.dumps(args,sort_keys=True)) in self.authorized,'write not explicitly authorized'
  visible='\n'.join(m['content'] for m in self.messages if m['role']=='user')
  if not fault:
   for key,value in args.items():
    if key.endswith(('_id','_ids')) or name.startswith('find_'):
     for v in value if isinstance(value,list) else [value]:assert str(v) in visible,f'unobserved argument: {key}={v}'
  before=self.tools.db.model_dump(mode='json')
  payload=json.dumps({'name':name,'arguments':args},separators=(',',':'),ensure_ascii=False)
  if malformed:payload=payload[:-2]
  text=note+'\n<tool_call>'+payload+'</tool_call>'
  index=self.add('assistant',text,target=not fault)
  try:
   calls=parse_calls(text);assert len(calls)==1
   validate(calls[0]['arguments'],self.schemas[name])
   result=getattr(self.tools,name)(**calls[0]['arguments']);body=result.model_dump(mode='json') if hasattr(result,'model_dump') else result
   error=False;response_name=name
  except Exception as e:
   if not fault:raise
   body={'error':str(e)};error=True;response_name='__runtime__' if malformed or type(e).__name__=='ValidationError' else name
  if fault:
   assert error,'intended fault did not fail';assert before==self.tools.db.model_dump(mode='json'),'fault changed DB'
   self.faults.append({'message_index':index,'type':fault,'loss_masked':True,'unchanged_database':True})
  else:assert not error
  self.add('user','<tool_response>'+json.dumps({'name':response_name,'content':body},ensure_ascii=False,separators=(',',':'))+'</tool_response>')
  self.audit.append({'message_index':index,'name':name,'arguments':args,'fault':fault,'error':error,'result':body})
  return body


def build(seed,n,notes):
 from tau2.domains.retail.utils import RETAIL_POLICY_PATH
 tools,user,orders,payment,cheaper=make_world(seed,n)
 names=['find_user_id_by_email','find_user_id_by_name_zip','get_user_details','get_order_details','get_product_details','cancel_pending_order','return_delivered_order_items','modify_pending_order_address','exchange_delivered_order_items','calculate']
 schemas=[t.openai_schema for t in tools.get_tools(include=names).values()]
 system='You are a retail support agent. Follow the policy. For each tool call, explain the immediate action in one concise sentence and emit exactly one complete <tool_call> JSON block. Inspect the result before any dependent call. Do not invent facts. Explain an error and correct its cause; never repeat an unchanged failing request. Track every unresolved customer request. Claim success only after the actual successful result.\n\n'+RETAIL_POLICY_PATH.read_text()+'\n\n<tools>'+json.dumps(schemas,separators=(',',':'))+'</tools>'
 b=Trace(tools,system,notes)
 names=[o.items[0].name for o in orders]
 destination={'address1':f'{470+n} Willow Lane','address2':f'Unit {12+n}','city':'Boston','state':'MA','country':'USA','zip':f'{2100+n:05d}'}
 b.add('user',f"I need to handle several orders. Cancel the orders containing {names[0]} and {names[1]} because I no longer need them. Return the delivered {names[2]} and {names[3]} using the original payment method. Keep the pending {names[4]} order, but change its shipping address. For the hiking boots, exchange only for an available cheaper pair of the same size and material; otherwise keep them. Please work through all of these and tell me the refund amounts.")
 bad=user.email.replace('@','-typo@')
 b.dialogue(notes['ask_email'],f'I think the email is {bad}.')
 b.call('find_user_id_by_email',{'email':bad},'I will try the email you supplied.',fault='customer_typo_lookup')
 b.dialogue(notes['email_retry'],f'I cannot remember the correct email. My first name is {user.name.first_name} and my last name is {user.name.last_name}.')
 b.dialogue(notes['ask_zip'],f'My account ZIP code is {user.address.zip}.')
 b.call('find_user_id_by_name_zip',{'first_name':user.name.first_name,'last_name':user.name.last_name,'zip':user.address.zip},'I will look up your account using the name and ZIP you supplied.')
 b.call('get_user_details',{'user_id':user.user_id},notes['read_account'],fault='incomplete_read_json',malformed=True)
 b.call('get_user_details',{'user_id':user.user_id},notes['repair_read'])
 for i,o in enumerate(orders):
  if i==2:b.call('get_order_details',{'order_number':o.order_id},'I will inspect the next order from the account list.',fault='wrong_argument_name')
  b.call('get_order_details',{'order_id':o.order_id},notes['repair_argument'] if i==2 else notes['read_order'])
 # Two cycles of status -> informed confirmation -> executed cancellation.
 previous=''
 for i in (0,1):
  o=orders[i];args={'order_id':o.order_id,'reason':'no longer needed'}
  b.confirm('cancel_pending_order',args,previous+f"Order {o.order_id} contains {o.items[0].name} and is pending. Shall I cancel it for 'no longer needed' and refund ${o.items[0].price:.2f} to its original payment method?")
  if i==0:b.call('cancel_pending_order',args,'I will submit the cancellation you confirmed.',fault='incomplete_write_json',malformed=True)
  result=b.call('cancel_pending_order',args,notes['repair_write'] if i==0 else 'I will cancel this pending order using your confirmation.')
  assert result['status']=='cancelled';previous=f"Order {o.order_id} is now cancelled, with a refund of ${o.items[0].price:.2f} recorded. "
 # Delivered orders require returns, and each original payment ID must be copied.
 for i in (2,3):
  o=orders[i]
  if i==2:b.call('cancel_pending_order',{'order_id':o.order_id,'reason':'no longer needed'},'I will cancel the delivered order.',fault='delivered_order_cannot_cancel')
  args={'order_id':o.order_id,'item_ids':[o.items[0].item_id],'payment_method_id':payment}
  lead=notes['delivered']+' ' if i==2 else previous
  b.confirm('return_delivered_order_items',args,lead+f"Order {o.order_id} contains {o.items[0].name}. May I request its return for ${o.items[0].price:.2f} using the original payment method {payment}? A return request still requires you to send the item back.")
  if i==3:b.call('return_delivered_order_items',{**args,'payment_method_id':payment+'-mistyped'},'I will submit the return request using the payment method.',fault='mistyped_payment_identifier')
  result=b.call('return_delivered_order_items',args,notes['repair_payment'] if i==3 else 'I will submit the return request you confirmed, using the item and payment identifiers from this order.')
  assert result['status']=='return requested';previous=f"The return of {o.items[0].name} from {o.order_id} is requested; it has not been refunded yet. "
 o=orders[4]
 b.dialogue(previous+f"Order {o.order_id} is pending and will be kept. What complete address should it use?",'Please use '+', '.join(destination.values())+'.')
 args={'order_id':o.order_id,**destination}
 b.confirm('modify_pending_order_address',args,f"Shall I change only order {o.order_id} to {destination['address1']}, {destination['address2']}, {destination['city']}, {destination['state']}, {destination['country']}, ZIP {destination['zip']}? Its total remains ${o.items[0].price:.2f}.")
 result=b.call('modify_pending_order_address',args,'I will apply the exact shipping address you confirmed, keeping the order active.')
 assert result['address']==destination
 o=orders[5];item=o.items[0]
 b.call('get_product_details',{'product_id':item.product_id},notes['catalog'],fault='late_incomplete_json',malformed=True)
 catalog=b.call('get_product_details',{'product_id':item.product_id},notes['repair_read']+' '+notes['catalog'])
 matches=[v for v in catalog['variants'].values() if v['available'] and v['options'].get('size')==item.options['size'] and v['options'].get('material')==item.options['material'] and Decimal(str(v['price']))<Decimal(str(item.price))]
 exchange_credit=Decimal('0')
 if matches:
  variant=min(matches,key=lambda v:v['price']);credit=round(item.price-variant['price'],2)
  b.call('calculate',{'expression':'!'},'I will calculate the exchange credit.',fault='late_invalid_calculation')
  amount=b.call('calculate',{'expression':f'{item.price:.2f} - {variant["price"]:.2f}'},notes['late_repair'])
  assert abs(float(amount)-credit)<.001
  args={'order_id':o.order_id,'item_ids':[item.item_id],'new_item_ids':[variant['item_id']],'payment_method_id':payment}
  b.confirm('exchange_delivered_order_items',args,f"The available replacement {variant['item_id']} is size {item.options['size']}, {item.options['material']}, and costs ${variant['price']:.2f}, compared with ${item.price:.2f} paid. The credit is ${credit:.2f}. Shall I request that exchange for order {o.order_id} using {payment}?")
  result=b.call('exchange_delivered_order_items',args,'I will request the exact same-size, same-material cheaper variant you confirmed.')
  assert result['status']=='exchange requested' and result['exchange_price_difference']==-credit
  exchange_credit=Decimal(str(credit));boots=f"Boot exchange requested for {o.order_id}; the price-difference credit is ${credit:.2f}."
 else:
  assert not cheaper;boots=f"Boot order {o.order_id} is unchanged: no available cheaper variant has both size {item.options['size']} and material {item.options['material']}."
 cancel_total=sum((Decimal(str(orders[i].items[0].price)) for i in (0,1)),Decimal('0'))
 return_total=sum((Decimal(str(orders[i].items[0].price)) for i in (2,3)),Decimal('0'))
 b.add('assistant',f"Completed: orders {orders[0].order_id} and {orders[1].order_id} are cancelled, and their recorded refunds total ${cancel_total:.2f}. Returns are requested for {orders[2].order_id} and {orders[3].order_id}; their combined item value is ${return_total:.2f}, pending the return process. Order {orders[4].order_id} remains pending at {destination['address1']}, {destination['address2']}, {destination['city']}, {destination['state']} {destination['zip']}, with its price unchanged at ${orders[4].items[0].price:.2f}. {boots} Return and exchange requests require the follow-up instructions; I am not claiming those conditional refunds have already been paid.")
 expected=['cancelled','cancelled','return requested','return requested','pending','exchange requested' if cheaper else 'delivered']
 assert [o.status for o in orders]==expected
 assert all(f['message_index'] not in b.targets for f in b.faults)
 return b,{'final_statuses':dict(zip([o.order_id for o in orders],expected)),'destination':destination,'cancellation_refunds':str(cancel_total),'conditional_return_value':str(return_total),'exchange_credit':str(exchange_credit),'all_goals_resolved':True}


def memory(prefix):
 facts=[];observed={};errors=[]
 for m in prefix[1:]:
  if m['role']!='user':continue
  if m['content'].startswith('<tool_response>'):
   body=json.loads(m['content'][15:-16]);content=body['content']
   if isinstance(content,dict) and 'error' in content:errors.append({'name':body['name'],'error':content['error'][:180]});continue
   key=body['name']
   if isinstance(content,dict):key+=':'+str(content.get('order_id',content.get('product_id',content.get('user_id',''))))
   observed[key]=content
  else:facts.append(m['content'])
 return {'customer_statements_and_confirmations':facts,'latest_successful_tool_observations':observed,'recent_tool_errors':errors[-3:],'instruction':'Only these observed facts are memory. Read the recent original messages for the current action; do not infer that a proposed action executed.'}


def export_windows(trace,tokenizer,family,split):
 rows=[]
 for target in trace.targets:
  prefix=trace.messages[:target];selected=trace.messages[:target+1];mode='full_history';tail_start=1
  count=lambda msgs:len(tokenizer.apply_chat_template(msgs,tokenize=True,return_dict=False))
  if count(selected)>8192:
   assistants=[i for i,m in enumerate(prefix) if m['role']=='assistant']
   for keep in (4,3,2,1,0):
    tail_start=assistants[-keep] if keep and len(assistants)>=keep else target
    # Memory ends before the verbatim tail and never contains target/future observations.
    mem=memory(trace.messages[:tail_start])
    context='Original customer request:\n'+trace.messages[1]['content']+'\n\nVerified working memory from earlier turns:\n<memory>'+json.dumps(mem,separators=(',',':'),ensure_ascii=False)+'</memory>'
    selected=[trace.messages[0],{'role':'user','content':context}]+trace.messages[tail_start:target+1]
    if count(selected)<=8192:mode='verified_memory_window';break
   else:raise ValueError('single target plus verified memory exceeds 8K')
  target_index=len(selected)-1
  encoded,_=tokenize_assistant_only(tokenizer,selected,8192,[target_index])
  assert len(encoded['input_ids'])<=8192 and sum(v!=-100 for v in encoded['labels'])>0
  # Every explicit ID in a supervised call must be sourced in visible user/tool context.
  visible='\n'.join(m['content'] for m in selected[:-1] if m['role']=='user')
  for call in parse_calls(selected[-1]['content']):
   for key,value in call['arguments'].items():
    if key.endswith(('_id','_ids')):
     for v in value if isinstance(value,list) else [value]:assert str(v) in visible,'window lost required identifier'
  rows.append({'id':family+f'-target-{target:03d}','task_id':family,'domain':'retail','group_id':digest({'domain':'retail','task_id':family}),'split':split,'messages':selected,'loss_message_indices':[target_index],
    'source_target_message_index':target,'source_tail_start':tail_start,'context_mode':mode,'context_tokens':len(encoded['input_ids']),'supervised_tokens':sum(v!=-100 for v in encoded['labels'])})
 assert len(rows)==len(trace.targets)
 return rows


def main():
 p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=12);p.add_argument('--output',default='outputs/retail-data/long-horizon-recovery-pilot');p.add_argument('--endpoint',default='http://127.0.0.1:8004/v1');p.add_argument('--model',default='qwen3.5-4b');p.add_argument('--tokenizer',default='outputs/hf-export/tiny-aya-global-function-calling-sft-525');p.add_argument('--deterministic-wording',action='store_true');a=p.parse_args()
 from tau2.domains.retail.data_model import get_db
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'records').mkdir(exist_ok=True);(out/'attempts').mkdir(exist_ok=True)
 seed=get_db();tokenizer=AutoTokenizer.from_pretrained(a.tokenizer);all_rows=[];records=[]
 atomic_json(out/'config.json',vars(a)|{'expert_plan':'deterministic, tool-executed','wording_model':None if a.deterministic_wording else a.model,'original_benchmark_tasks_used':False})
 with OpenAI(base_url=a.endpoint,api_key='local',timeout=180,max_retries=2) as client:
  for n in range(a.count):
   file=out/'records'/f'{n:03d}.json'
   if file.exists():
    record=json.loads(file.read_text());trace=type('SavedTrace',(),{'messages':record['messages'],'targets':record['loss_message_indices']})();rows=export_windows(trace,tokenizer,record['id'],record['split'])
   else:
    notes={k:v[0] for k,v in NOTES.items()} if a.deterministic_wording else wording(client,a.model,out,n)
    b,checks=build(seed,n,notes);family=f'long-horizon-recovery-{n:04d}';split='validation' if n%10==9 else 'train'
    rows=export_windows(b,tokenizer,family,split)
    record={'id':family,'task_id':family,'split':split,'messages':b.messages,'loss_message_indices':b.targets,'masked_faults':b.faults,'tool_execution_audit':b.audit,'checks':checks,
      'assistant_turns':sum(m['role']=='assistant' for m in b.messages),'correct_tool_calls':sum(not x['error'] for x in b.audit),'full_context_tokens':len(tokenizer.apply_chat_template(b.messages,tokenize=True,return_dict=False)),
      'wording_model':None if a.deterministic_wording else a.model,'synthetic_expert_plan':True}
    atomic_json(file,record)
   records.append(record);all_rows.extend(rows)
   file.with_suffix('.txt').write_text('\n\n'.join(('ASSISTANT [MASKED FAULT — CONTEXT ONLY]' if i in {f['message_index'] for f in record['masked_faults']} else ('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper()))+': '+m['content'] for i,m in enumerate(record['messages']) if i))
   for name,split in [('sft','train'),('validation','validation'),('all',None)]:
    dest=out/(name+'.jsonl');temp=dest.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows if split is None or r['split']==split));temp.replace(dest)
   report={'completed':len(records),'target':a.count,'complete':len(records)==a.count,'training_windows':sum(r['split']=='train' for r in all_rows),'validation_windows':sum(r['split']=='validation' for r in all_rows),'max_window_tokens':max(r['context_tokens'] for r in all_rows),'windowed_targets':sum(r['context_mode']=='verified_memory_window' for r in all_rows),'assistant_turns':[r['assistant_turns'] for r in records],'masked_faults':sum(len(r['masked_faults']) for r in records)}
   atomic_json(out/'task-split-manifest.json',{'partitions':{
     'sft':[r['task_id'] for r in records if r['split']=='train'],
     'validation':[r['task_id'] for r in records if r['split']=='validation']},
     'split_unit':'synthetic_family','benchmark_task_ids_used':False})
   atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)
 (out/'conversations.txt').write_text('\n\n'.join('===== '+r['id']+' =====\n'+(out/'records'/f'{i:03d}.txt').read_text() for i,r in enumerate(records)))
if __name__=='__main__':main()
