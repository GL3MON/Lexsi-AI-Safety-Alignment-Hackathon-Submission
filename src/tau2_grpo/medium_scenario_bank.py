"""Medium workflows: actors must generate the full route, not just its suffix."""
import copy,json,re
from tau2_grpo.short_scenario_bank import BankEnvironment,body,call,fingerprint,tools_from_world
FAMILIES=('status_email','status_missing_zip','cancel_email','cancel_missing_email','change_shipping','change_payment','return_item','exchange_item')
def build_medium_case(world,catalog,system,family):
 tools=tools_from_world(world,catalog);u=next(iter(tools.db.users.values()));pending=next(o for o in tools.db.orders.values() if o.status=='pending');delivered=next(o for o in tools.db.orders.values() if o.status=='delivered')
 lookup=call('find_user_id_by_email',{'email':u.email});profile=call('get_user_details',{'user_id':u.user_id});order=call('get_order_details',{'order_id':pending.order_id});prefix=f'My email is {u.email}. '
 route=[lookup,profile,order]
 if family=='status_missing_zip':
  prefix=f'My first name is {u.name.first_name} and last name is {u.name.last_name}. I cannot remember my email. '
  route=[dict(kind='ask',field='zip',reply=f'My ZIP code is {u.address.zip}.'),call('find_user_id_by_name_zip',dict(first_name=u.name.first_name,last_name=u.name.last_name,zip=u.address.zip)),profile,order]
 if family.startswith('status_'):
  request=f'Please locate my account, retrieve my profile, and check the current status of order {pending.order_id}. Do not change anything.'
  route.append(dict(kind='status',order_id=pending.order_id,status=pending.status))
 elif family.startswith('cancel_'):
  if family=='cancel_missing_email':
   prefix=f'My name is {u.name.first_name} {u.name.last_name}. Please find my account using email. '
   route=[dict(kind='ask',field='email',reply=f'My email is {u.email}.')]+route
  request=f'Please verify my account and profile, check order {pending.order_id}, then cancel the entire order because it is no longer needed. Ask for my confirmation before changing anything.'
  route += [dict(kind='medium_confirm',action='cancel',order_id=pending.order_id,reply=f'Yes, cancel the entire order {pending.order_id}; it is no longer needed.'),call('cancel_pending_order',dict(order_id=pending.order_id,reason='no longer needed'))]
 elif family=='change_shipping':
  address=dict(address1='705 Linden Avenue',address2='Unit 14',city='Madison',state='WI',country='USA',zip='53703')
  request=f'Verify my account and profile, retrieve order {pending.order_id}, then change its shipping address to {json.dumps(address)}. Ask for confirmation first.'
  route += [dict(kind='medium_confirm',action='address',order_id=pending.order_id,reply='Yes, change the shipping address to the exact address I provided.'),call('modify_pending_order_address',dict(order_id=pending.order_id,**address))]
 elif family=='change_payment':
  request=f'Verify my account and profile, retrieve order {pending.order_id}, then change its payment to saved method {world["alternate_payment"]}. Ask for confirmation first.'
  route += [dict(kind='medium_confirm',action='payment',order_id=pending.order_id,reply=f'Yes, change the payment to {world["alternate_payment"]}.'),call('modify_pending_order_payment',dict(order_id=pending.order_id,payment_method_id=world['alternate_payment']))]
 elif family=='return_item':
  route=[lookup,profile,call('get_order_details',dict(order_id=delivered.order_id))]
  item=delivered.items[0];request=f'Verify my account and profile, retrieve delivered order {delivered.order_id}, then return item {item.item_id}, refunding its {item.price} price to original payment method {world["primary_payment"]}. Ask for confirmation first.'
  route += [dict(kind='medium_confirm',action='return',order_id=delivered.order_id,reply='Yes, return that item and refund the original payment method.'),call('return_delivered_order_items',dict(order_id=delivered.order_id,item_ids=[item.item_id],payment_method_id=world['primary_payment']))]
 else:
  item=delivered.items[0];variant=next(v for v in tools.db.products[item.product_id].variants.values() if v.available and v.item_id!=item.item_id)
  route=[lookup,profile,call('get_order_details',dict(order_id=delivered.order_id)),call('get_product_details',dict(product_id=item.product_id)),call('calculate',dict(expression=f'{variant.price} - {item.price}'),match='math')]
  request=f'Verify my account and profile, retrieve delivered order {delivered.order_id} and product {item.product_id}. Exchange item {item.item_id} for variant {variant.item_id} of the same product. Use calculate to check the new variant price minus the current item price. Use saved payment method {world["alternate_payment"]} for the difference and ask me for confirmation before executing.'
  route += [dict(kind='medium_confirm',action='exchange',order_id=delivered.order_id,reply='Yes, exchange those exact items using the saved payment method specified.'),call('exchange_delivered_order_items',dict(order_id=delivered.order_id,item_ids=[item.item_id],new_item_ids=[variant.item_id],payment_method_id=world['alternate_payment']))]
 return dict(id=f'{world["world_id"]}-medium-{family}',world_id=world['world_id'],family=family,messages=[dict(role='system',content=system),dict(role='user',content=prefix+request)],routes=[route],max_actor_turns=min(8,len(route)+1),reference_actor_turns=len(route),response_tokens=192,context_limit=8192,synthetic_curriculum=True,native_tau2_task=False,recoverable_error_budget=1)

class MediumEnvironment(BankEnvironment):
 """Keep a bounded repair opportunity for objectively invalid tool syntax/schema."""
 def reset(self):self.repairs=0;return super().reset()
 def step(self,text):
  from tau2_grpo.prepare_function_calling import parse_calls
  from tau2_grpo.short_scenario_bank import matches
  import jsonschema
  error=None
  try:
   parsed=parse_calls(text)
   for c in parsed:
    if c['name'] not in self.schemas:raise ValueError('Unknown tool: '+c['name']+'. Use a tool from the system schema.')
    jsonschema.validate(c['arguments'],self.schemas[c['name']])
  except (ValueError,jsonschema.ValidationError) as exc:error=str(exc)[:600]
  if error and '<tool_call>' in text and self.repairs<self.case.get('recoverable_error_budget',0):
   self.repairs+=1;incoming=[dict(role='tool',name='__runtime__',error=True,content=json.dumps(dict(name='__runtime__',content={'error':error})))];self.history.append(dict(role='assistant',content=text));self.history.extend(copy.deepcopy(incoming));self.checks.append(dict(criterion='recoverable_invalid_call',passed=False,error=error));return incoming,False,None
  if self.routes and self.routes[0][0]['kind']=='medium_confirm':
   step=self.routes[0][0];lower=text.lower()
   terms={'cancel':r'cancel','address':r'address|shipping','payment':r'payment','return':r'return|refund','exchange':r'exchange|replac'}
   ok=not error and not parsed and bool(re.search(r'confirm|proceed|shall i|may i|would you like',lower)) and bool(re.search(terms[step['action']],lower)) and not re.search(r'i (?:have |already )?(?:cancelled|changed|returned|exchanged)|successfully',lower)
   visible='\n'.join(m['content'] for m in self.history if m['role'] in ('user','tool'))
   if any(v not in visible for v in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d[\d.,]*\b',text)):ok=False
   if ok:
    self.routes=[r[1:] for r in self.routes];incoming=[dict(role='user',content=step['reply'])];self.history.append(dict(role='assistant',content=text));self.history.extend(copy.deepcopy(incoming));self.checks.append(dict(criterion='confirmation_before_write',passed=True));return incoming,False,None
  return super().step(text)


 def finish(self):
  # Verified substeps provide learning signal when whole-workflow success is rare.
  # Customer/tool text is never a target; only environment-checked actions count.
  total=len(self.case['routes'][0]);passed=sum(c['passed'] for c in self.checks)
  progress=min(passed,total)/total
  reward=1.0 if self.success else min(0.99,2*progress-1)
  self.reward_details=dict(version='retail-medium-progress-v1',family=self.case['family'],short_success=self.success,verified_steps=passed,required_steps=total,progress=progress,total=reward,checks=self.checks)
  return reward,dict(reward_info=dict(reward_basis=['VERIFIED_WORKFLOW_PROGRESS'],short_checks=self.checks),short_success=self.success,synthetic_curriculum=True,native_tau2_task=False)


def reference_text(step):
 if step['kind']=='call':return '<tool_call>'+json.dumps(dict(name=step['name'],arguments=step['arguments']),separators=(',',':'))+'</tool_call>'
 if step['kind']=='ask':return 'Could you please provide your '+('email address' if step['field']=='email' else 'ZIP code')+'?'
 if step['kind']=='medium_confirm':return f'Please confirm: may I proceed with the {step["action"]} change for order {step["order_id"]}?'
 return f'Order {step["order_id"]} is {step["status"]}.'

def validate_medium(case,world,catalog,tokenizer):
 env=MediumEnvironment(case,world,catalog);history=env.reset();peak=0
 for i,step in enumerate(case['routes'][0]):
  n=len(tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,return_dict=False));peak=max(peak,n)
  # Reserve all remaining actions plus one full response for possible recovery.
  if n+(len(case['routes'][0])-i+1)*case['response_tokens']>8192:raise ValueError('Medium workflow exceeds 8K including response reserve')
  text=reference_text(step);incoming,done,_=env.step(text);history.append(dict(role='assistant',content=text));history.extend(incoming)
  if env.done and not env.success:raise ValueError('Medium reference failed: '+str(env.checks))
 if not env.success:raise ValueError('Medium reference incomplete')
 case['expected_state_hashes']=[fingerprint(env.tools.db)];case['max_reference_prompt_tokens']=peak;return case
