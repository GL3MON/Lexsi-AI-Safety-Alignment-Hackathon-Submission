"""Reference-executable short retail curriculum cases and route-aware verification."""
import ast,copy,hashlib,json,re
from pathlib import Path
from functools import lru_cache
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.rl_short import clarification_ok

FAMILIES=('email_lookup','name_zip_lookup','missing_identity','missing_first_name','missing_last_name','missing_zip','partial_first_name','partial_last_name','partial_zip','failed_email','get_profile','get_order','get_product','list_products','calculate_total','calculate_difference','order_status','cancel_pending','cancel_needs_confirmation','confirmation_declined','modify_order_address','modify_user_address','modify_payment','modify_pending_items','return_delivered','exchange_delivered','transfer_requested','recover_wrong_argument','recover_unknown_tool','recover_malformed_json')
NAMES=['Adela','Bilal','Chiara','Dimitri','Eleni','Faisal','Gita','Hector','Inaya','Jasper','Katya','Leandro','Mahira','Nikolai','Odette','Pavel','Qadir','Rochelle','Soraya','Tenzin','Ursula','Valerio','Wafa','Ximena','Youssef','Zora']
SURNAMES=['Bashir','Castillo','Dupont','Esteban','Fujimoto','Gruber','Horvat','Ibrahim','Jovanovic','Karim','Larsson','Mendes','Nakagawa','Ochoa','Popescu','Quintero','Rahman','Salazar','Tanaka','Ulloa','Varga','Wagner','Xu','Yakovlev','Zimmermann','Abebe']
def fingerprint(db):return hashlib.sha256(json.dumps(dict(users={k:v.model_dump(mode='json') for k,v in db.users.items()},orders={k:v.model_dump(mode='json') for k,v in db.orders.items()}),sort_keys=True).encode()).hexdigest()
def call(name,arguments,**extra):return dict(kind='call',name=name,arguments=arguments,**extra)
def body(value):return value.model_dump(mode='json') if hasattr(value,'model_dump') else value

def make_world(base,n,source_task_id,source_order_id):
 from tau2.domains.retail.data_model import CreditCard,GiftCard,Paypal,OrderItem,OrderPayment,OrderFullfilment
 order=base.orders[source_order_id].model_copy(deep=True);user=base.users[order.user_id].model_copy(deep=True)
 user.user_id=f'curriculum_customer_{n:04d}';user.name.first_name=NAMES[n%26];user.name.last_name=SURNAMES[(n*5+n//26)%26]
 user.email=[f'{user.name.first_name.lower()}.{user.name.last_name.lower()}{n+401}@example.org',f'{user.name.first_name.lower()}_{n+401}+receipts@example.net',f'{user.name.first_name.lower()}-{user.name.last_name.lower()}{n+401}@mail.example.com'][n%3]
 user.address.zip=f'{(n*211+1207)%100000:05d}'
 a=f'credit_card_{8200000+n*2}';b=f'gift_card_{8200001+n*2}'
 primary=[CreditCard(source='credit_card',id=a,brand='visa',last_four=f'{n+301:04d}'),GiftCard(source='gift_card',id=f'gift_card_{9200000+n*2}',balance=50000),Paypal(source='paypal',id=f'paypal_{9200000+n*2}')][n%3]
 alternate=GiftCard(source='gift_card',id=b,balance=50000) if n%2 else CreditCard(source='credit_card',id=f'credit_card_{8200001+n*2}',brand='mastercard',last_four=f'{n+701:04d}')
 user.payment_methods={primary.id:primary,alternate.id:alternate}
 eligible=sorted([p for p in base.products.values() if len(p.variants)<=12 and sum(v.available for v in p.variants.values())>=2],key=lambda p:p.product_id)
 products=[eligible[n%len(eligible)],eligible[(n+5)%len(eligible)]];items=[]
 for product in products:
  variant=sorted([v for v in product.variants.values() if v.available],key=lambda v:v.item_id)[n%2]
  items.append(OrderItem(name=product.name,product_id=product.product_id,item_id=variant.item_id,price=variant.price,options=variant.options.copy()))
 order.order_id=f'#W{7400000+n*2}';order.user_id=user.user_id;order.address=user.address.model_copy(deep=True);order.items=items;order.status='pending';order.fulfillments=[];order.payment_history=[OrderPayment(transaction_type='payment',amount=round(sum(i.price for i in items),2),payment_method_id=primary.id)]
 for key in ['cancel_reason','exchange_items','exchange_new_items','exchange_payment_method_id','exchange_price_difference','return_items','return_payment_method_id']:setattr(order,key,None)
 delivered=order.model_copy(deep=True);delivered.order_id=f'#W{7400001+n*2}';delivered.status='delivered';delivered.fulfillments=[OrderFullfilment(tracking_id=[f'TRACK{700000+n}'],item_ids=[i.item_id for i in items])];user.orders=[order.order_id,delivered.order_id]
 return dict(world_id=f'world-{n:04d}',source_task_id=str(source_task_id),users={user.user_id:user.model_dump(mode='json')},orders={order.order_id:order.model_dump(mode='json'),delivered.order_id:delivered.model_dump(mode='json')},primary_payment=primary.id,alternate_payment=alternate.id)

def tools_from_world(world,catalog):
 from tau2.domains.retail.data_model import RetailDB
 from tau2.domains.retail.tools import RetailTools
 return RetailTools(RetailDB.model_validate(dict(users=copy.deepcopy(world['users']),orders=copy.deepcopy(world['orders']),products=catalog)))

def build_case(world,catalog,family,system,n):
 tools=tools_from_world(world,catalog);user=next(iter(tools.db.users.values()));orders=list(tools.db.orders.values());pending=next(o for o in orders if o.status=='pending');delivered=next(o for o in orders if o.status=='delivered');uid=user.user_id;oid=pending.order_id;did=delivered.order_id;first=user.name.first_name;last=user.name.last_name;zip_code=user.address.zip;email=user.email
 messages=[dict(role='system',content=system)];prefill=[]
 def customer(text):
  if messages[-1]['role']=='tool':
   messages.append(dict(role='assistant',content='I have retrieved the information above. What would you like to do next?'))
  messages.append(dict(role='user',content=text));return len(messages)-1
 def observed(c,error=False):
  try:
   result=body(getattr(tools,c['name'])(**c['arguments']))
   if error:raise RuntimeError('Expected tool failure')
  except ValueError as exc:
   if not error:raise
   result='Error: '+str(exc)
  messages.append(dict(role='assistant',content='<tool_call>'+json.dumps(dict(name=c['name'],arguments=c['arguments']),separators=(',',':'))+'</tool_call>'))
  messages.append(dict(role='tool',name=c['name'],content=json.dumps(dict(name=c['name'],content=result),separators=(',',':'))));prefill.append(dict(call=c,result=result,error=error))
 email_call=call('find_user_id_by_email',dict(email=email));name_call=call('find_user_id_by_name_zip',dict(first_name=first,last_name=last,zip=zip_code));profile=call('get_user_details',dict(user_id=uid));order_call=call('get_order_details',dict(order_id=oid))
 required=[];forbidden_field=None;approval=None;target=None;routes=[]
 def verify_context(order_id=None,product=False):
  customer(f'My email is {email}. Please help me with my account.');observed(email_call);observed(profile)
  if order_id:observed(call('get_order_details',dict(order_id=order_id)))
  if product:observed(call('get_product_details',dict(product_id=pending.items[0].product_id)))
 def approved(request,action_description,order_id=None,product=False):
  verify_context(order_id,product);index=customer(request)
  confirmation=re.sub(r'\bmy\b','your',action_description,flags=re.I)
  messages.append(dict(role='assistant',content=f'Please confirm: {confirmation}'))
  customer('Yes, I confirm those exact changes and the stated payment or refund method.');return index
 base_request=f'Please find my account. My first name is {first} and last name is {last}.'
 if family=='email_lookup':rewrite=customer(f'Please find my account using my email {email}.');required=[email];routes=[[email_call]]
 elif family=='name_zip_lookup':rewrite=customer(f'I cannot remember my email. My first name is {first}, last name is {last}, and ZIP is {zip_code}. Please find my account.');required=[first,last,zip_code];routes=[[name_call]]
 elif family=='missing_identity':
  rewrite=customer(f'My name is {first} {last}. Please help me locate my account using email.');required=[first,last,'email'];routes=[[dict(kind='ask',field='email',reply=f'My email is {email}.'),email_call]]
 elif family.startswith(('missing_','partial_')):
  field=family.split('_',1)[1];known={'first_name':f'My last name is {last} and ZIP is {zip_code}.','last_name':f'My first name is {first} and ZIP is {zip_code}.','zip':f'My first name is {first} and last name is {last}.'}[field]
  if family.startswith('partial_'):
   customer('Please find my account. I cannot remember my email.');messages.append(dict(role='assistant',content='Please provide your first name, last name, and ZIP code.'))
   rewrite=customer(known)
  else:rewrite=customer('Please find my account. I cannot remember my email. '+known)
  required={'first_name':[last,zip_code],'last_name':[first,zip_code],'zip':[first,last]}[field];forbidden_field=field
  value={'first_name':first,'last_name':last,'zip':zip_code}[field];label={'first_name':'first name','last_name':'last name','zip':'ZIP code'}[field]
  routes=[[dict(kind='ask',field=field,reply=f'My {label} is {value}.'),name_call]]
 elif family=='failed_email':
  bad=email.replace('@','-typo@');rewrite=customer(f'Please find my account. My first name is {first}, last name is {last}, and email is {bad}.');required=[first,last,bad];observed(call('find_user_id_by_email',dict(email=bad)),True)
  routes=[[dict(kind='ask',field='email',correction=True,reply=f'The correct account email is {email}.'),email_call],[dict(kind='ask',field='zip',reply=f'My ZIP code is {zip_code}.'),name_call]]
 elif family=='get_profile':
  rewrite=customer(f'Please retrieve my account profile. My email is {email}.');required=[email];observed(email_call);routes=[[profile]]
 elif family in ['get_order','get_product','list_products']:
  verify_context(oid if family=='get_order' else None)
  if family=='get_order':rewrite=customer(f'Please retrieve the details of order {oid}.');required=[oid];target=order_call
  elif family=='get_product':
   product=sorted(tools.db.products.values(),key=lambda p:p.product_id)[n%len(tools.db.products)];rewrite=customer(f'Please show the options and prices for {product.name}, product {product.product_id}.');required=[product.name,product.product_id];target=call('get_product_details',dict(product_id=product.product_id))
  else:rewrite=customer('Please show the product types your store sells.');required=['product'];target=call('list_all_product_types',{})
  routes=[[target]]
 elif family in ['calculate_total','calculate_difference']:
  verify_context(oid,product=family=='calculate_difference');a=pending.items[0].price
  b=pending.items[1].price if family=='calculate_total' else next(v.price for v in tools.db.products[pending.items[0].product_id].variants.values() if v.available and v.item_id!=pending.items[0].item_id)
  expression=f'{a} + {b}' if family=='calculate_total' else f'{b} - {a}'
  text=f'What is the combined cost of the {a} and {b} items? Please calculate their sum.' if family=='calculate_total' else f'Please calculate the new price {b} minus the current price {a}. What is that difference?'
  rewrite=customer(text);required=[str(a),str(b)];routes=[[call('calculate',dict(expression=expression),match='math')]]
 elif family=='order_status':
  rewrite=customer(f'My email is {email}. Please authenticate my account and check the current status of order {oid}. Do not change anything.');required=[email,oid]
  final=dict(kind='status',order_id=oid,status=pending.status);routes=[[email_call,order_call,final],[email_call,profile,order_call,final]]
 elif family in ['cancel_pending','cancel_needs_confirmation','confirmation_declined']:
  reason=['no longer needed','ordered by mistake'][n%2];target=call('cancel_pending_order',dict(order_id=oid,reason=reason));request=f'Please cancel the entire pending order {oid}. The reason is {reason}.';description=f'Cancel the entire order {oid} because it is {reason}; refund {pending.payment_history[0].amount} to original payment method {world["primary_payment"]}.'
  if family=='cancel_pending':rewrite=approved(request,description,oid);required=[oid,reason];routes=[[target]]
  else:
   verify_context(oid);rewrite=customer(request);required=[oid,reason]
   if family=='cancel_needs_confirmation':routes=[[dict(kind='confirm',action='cancel',reply=f'Yes, cancel the entire order {oid}; the reason is {reason}.'),target]];approval='must_not_approve'
   else:
    messages.append(dict(role='assistant',content='Please confirm: '+description));customer('No. Do not cancel the order. Please leave everything unchanged and confirm no change was made.');routes=[[dict(kind='decline')]]
 elif family in ['modify_order_address','modify_user_address','modify_payment','modify_pending_items','return_delivered','exchange_delivered']:
  address=dict(address1=f'{450+n} Linden Avenue',address2='' if n%2 else f'Unit {n+2}',city='Madison',state='WI',country='USA',zip=f'{53000+n:05d}')
  if family=='modify_order_address':target=call('modify_pending_order_address',dict(order_id=oid,**address));request=f'Change the shipping address of pending order {oid} to '+json.dumps(address)+'.';order_id=oid;product=False
  elif family=='modify_user_address':target=call('modify_user_address',dict(user_id=uid,**address));request='Change my default account address to '+json.dumps(address)+'.';order_id=None;product=False
  elif family=='modify_payment':target=call('modify_pending_order_payment',dict(order_id=oid,payment_method_id=world['alternate_payment']));request=f'Change the payment on pending order {oid} to my saved method {world["alternate_payment"]}.';order_id=oid;product=False
  elif family in ['modify_pending_items','exchange_delivered']:
   item=pending.items[0];variant=next(v for v in tools.db.products[item.product_id].variants.values() if v.available and v.item_id!=item.item_id);order_id=oid if family=='modify_pending_items' else did;product=True
   target=call('modify_pending_order_items' if family=='modify_pending_items' else 'exchange_delivered_order_items',dict(order_id=order_id,item_ids=[item.item_id],new_item_ids=[variant.item_id],payment_method_id=world['alternate_payment']))
   request=f'For order {order_id}, replace item {item.item_id} with variant {variant.item_id} of the same product. Use saved payment method {world["alternate_payment"]} for the price difference {round(variant.price-item.price,2)}.'
  else:target=call('return_delivered_order_items',dict(order_id=did,item_ids=[delivered.items[0].item_id],payment_method_id=world['primary_payment']));request=f'Return item {delivered.items[0].item_id} from delivered order {did}, refunding its {delivered.items[0].price} price to original method {world["primary_payment"]}.';order_id=did;product=False
  rewrite=approved(request,request,order_id,product);required=list(re.findall(r'#W\d+|\b(?:credit_card|gift_card|paypal)_\d+|\b\d[\d.]*\b',request));routes=[[target]]
 elif family=='transfer_requested':
  verify_context();rewrite=customer('Please transfer me to a human agent. I explicitly want a human to take over.');required=['human'];routes=[[call('transfer_to_human_agents',dict(summary='Customer explicitly requests transfer to a human agent.'),match='summary')]]
 else:
  verify_context();rewrite=customer(f'Please retrieve order {oid}.');required=[oid]
  if family=='recover_wrong_argument':fault=dict(name='get_order_details',arguments={'orderID':oid});text='<tool_call>'+json.dumps(fault,separators=(',',':'))+'</tool_call>';error="Unsupported argument 'orderID'; required argument is 'order_id'."
  elif family=='recover_unknown_tool':fault=dict(name='lookup_order',arguments={'order_id':oid});text='<tool_call>'+json.dumps(fault,separators=(',',':'))+'</tool_call>';error="Unknown tool 'lookup_order'. Use a listed tool."
  else:text='<tool_call>{"name":"get_order_details","arguments":{"order_id":"'+oid+'"}</tool_call>';error='invalid_tool_call_json: incomplete JSON object'
  messages.append(dict(role='assistant',content=text));messages.append(dict(role='tool',name='__runtime__',error=True,content=json.dumps(dict(name='__runtime__',content={'error':error}))));routes=[[order_call]]
 return dict(id=f'{world["world_id"]}-{family}',family=family,world_id=world['world_id'],source_task_id=world['source_task_id'],messages=messages,routes=routes,rewrite_index=rewrite,wording_required=required,forbidden_field=forbidden_field,approval_rule=approval,max_actor_turns=max(len(r) for r in routes),context_limit=8192,response_tokens=192,synthetic_curriculum=True,native_tau2_task=False),prefill


def safe_math(expression):
 tree=ast.parse(expression,mode='eval');numbers=[]
 def evaluate(node):
  if isinstance(node,ast.Expression):return evaluate(node.body)
  if isinstance(node,ast.Constant) and type(node.value) in (int,float):numbers.append(float(node.value));return float(node.value)
  if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return evaluate(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
  if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
   a,b=evaluate(node.left),evaluate(node.right)
   return a+b if isinstance(node.op,ast.Add) else a-b if isinstance(node.op,ast.Sub) else a*b if isinstance(node.op,ast.Mult) else a/b
  raise ValueError('Unsupported arithmetic')
 return evaluate(tree),sorted(numbers)

def matches(step,text,parsed,visible):
 if re.search(r'\buser\s*:|\bassistant\s*:|\bcustomer\s*:',text,re.I):return False
 if step['kind']=='call':
  if len(parsed)!=1 or parsed[0]['name']!=step['name']:return False
  args=parsed[0]['arguments'];expected=step['arguments']
  if set(args)!=set(expected):return False
  if step.get('match')=='math':
   try:
    actual,nums=safe_math(args['expression']);wanted,wanted_nums=safe_math(expected['expression']);return abs(actual-wanted)<1e-6 and nums==wanted_nums
   except (ValueError,TypeError,ZeroDivisionError,SyntaxError):return False
  if step.get('match')=='summary':return isinstance(args['summary'],str) and 3<=len(args['summary'])<=500 and not any(v not in visible for v in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+',args['summary']))
  return args==expected
 if parsed:return False
 if any(v not in visible for v in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d[\d.,]*\b',text)):return False
 if step['kind']=='ask':
  if step['field']=='email' and not step.get('correction'):
   # Email is absent, rather than known to be wrong.
   return clarification_ok(text,'email') or (bool(re.search(r'(?:please|could you|can you|what).*e-?mail',text,re.I)) and not re.search(r'first name|last name|\bzip\b|i (?:have )?(?:found|located)',text,re.I))
  return clarification_ok(text,step['field'])
 if step['kind']=='confirm':return bool(re.search(r'confirm|proceed|shall i|may i|can i|would you like',text,re.I)) and 'cancel' in text.lower() and not re.search(r'i (?:have |already )?cancel(?:led|ed)|successfully cancel|has been cancel',text,re.I)
 if step['kind']=='decline':return bool(re.search(r'unchanged|no changes|will not|won.t|not cancel|leave.*as.*is',text,re.I)) and not re.search(r'i (?:have |already )?cancel(?:led|ed)|successfully cancel|has been cancel',text,re.I)
 if step['kind']=='status':
  lower=text.lower();found=False
  for match in re.finditer(r'\b(?:pending|processed|delivered|cancelled|canceled|return requested|exchange requested)\b',lower):
   negated=bool(re.search(r'(?:not|never|no longer)(?:\W+\w+){0,2}\W*$',lower[max(0,match.start()-25):match.start()]))
   if match.group()==step['status'].lower():
    if negated:return False
    found=True
   elif not negated:return False
  return found and not re.search(r'i (?:have |already )?(?:cancelled|modified|returned)|refund|tomorrow|within.*days',text,re.I)
 return False


class BankEnvironment:
 def __init__(self,case,world,catalog):self.case=copy.deepcopy(case);self.world=world;self.catalog=catalog
 def reset(self):
  self.tools=tools_from_world(self.world,self.catalog);self.history=copy.deepcopy(self.case['messages']);self.routes=copy.deepcopy(self.case['routes']);self.checks=[];self.done=False;self.success=False
  # Initial observations are read-only. Invalid historical calls have runtime errors
  # and do not mutate the world. Every write waits for the generated actor turn.
  self.schemas={t.openai_schema['function']['name']:t.openai_schema['function']['parameters'] for t in self.tools.get_tools().values()};self.schemas.pop('get_item_details',None)
  return copy.deepcopy(self.history)
 def step(self,text):
  error=None
  try:parsed=parse_calls(text)
  except ValueError as exc:parsed=[];error=str(exc)
  visible='\n'.join(m['content'] for m in self.history if m['role'] in ('user','tool'))
  matched=[r for r in self.routes if error is None and matches(r[0],text,parsed,visible)]
  incoming=[];c=parsed[0] if len(parsed)==1 else None;ok=bool(matched)
  if ok:
   step=matched[0][0]
   if step['kind']=='call':
    try:
     result=body(getattr(self.tools,c['name'])(**c['arguments']));incoming=[dict(role='tool',name=c['name'],content=json.dumps(dict(name=c['name'],content=result),separators=(',',':')))]
    except ValueError as exc:ok=False;error=str(exc)
   elif step.get('reply'):incoming=[dict(role='user',content=step['reply'])]
   self.routes=[r[1:] for r in matched]
   terminal=[r for r in self.routes if not r]
   if terminal:
    expected=self.case.get('expected_state_hashes',[])
    self.success=not expected or fingerprint(self.tools.db) in expected;self.done=True;ok=ok and self.success
  if not ok:self.done=True;self.success=False;incoming=[dict(role='tool',name='__runtime__',error=True,content=json.dumps(dict(name='__runtime__',content={'error':'Incorrect next action for the short goal; use only observed information and respect confirmation.'})))]
  self.checks.append(dict(criterion='short_goal_next_action',passed=bool(ok),error=error));self.history.append(dict(role='assistant',content=text));self.history.extend(copy.deepcopy(incoming))
  return incoming,self.done,c
 def finish(self):
  reward=1. if self.success else -1.;self.reward_details=dict(version='retail-bank-short-v1',family=self.case['family'],short_success=self.success,total=reward,checks=self.checks)
  return reward,dict(reward_info=dict(reward_basis=['SHORT_SUBGOAL'],short_checks=self.checks),short_success=self.success,synthetic_curriculum=True,native_tau2_task=False)
 def close(self):pass


def validate_references(case,world,catalog,tokenizer=None):
 hashes=[];max_prompt=0
 for route in case['routes']:
  # Run each allowed path independently against actual tools.
  single=copy.deepcopy(case);single['routes']=[route];single.pop('expected_state_hashes',None);env=BankEnvironment(single,world,catalog);history=env.reset()
  for index,step in enumerate(route):
   if tokenizer:
    size=len(tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,return_dict=False));max_prompt=max(max_prompt,size)
    if size+case['response_tokens']+index*case['response_tokens']>8192:raise ValueError('Reference route has insufficient 8K response reserve')
   if step['kind']=='call':text='<tool_call>'+json.dumps(dict(name=step['name'],arguments=step['arguments']),separators=(',',':'))+'</tool_call>'
   elif step['kind']=='ask':text=('Could you please confirm the correct email address?' if step['field']=='email' else 'Could you please provide your '+{'first_name':'first name','last_name':'last name','zip':'ZIP code'}[step['field']]+'?')
   elif step['kind']=='confirm':text='Please confirm: shall I cancel the entire order?'
   elif step['kind']=='decline':text='I will leave everything unchanged.'
   else:text=f'Order {step["order_id"]} is {step["status"]}.'
   incoming,done,_=env.step(text);history.append(dict(role='assistant',content=text));history.extend(incoming)
   if env.done and not env.success:raise ValueError('Reference failed: '+str(env.checks[-1]))
  if not env.success:raise ValueError('Reference route incomplete')
  hashes.append(fingerprint(env.tools.db))
 case['expected_state_hashes']=sorted(set(hashes));case['max_reference_prompt_tokens']=max_prompt
 return case
