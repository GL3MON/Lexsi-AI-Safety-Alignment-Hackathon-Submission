"""Read-only assistance executed in the real Tau2 environment, never policy targets."""
import copy,json,re
from tau2_grpo.rl_rewards import tool_body,failed,observed
READ_ONLY={'find_user_id_by_email','find_user_id_by_name_zip','get_user_details','get_order_details','get_product_details','list_all_product_types','calculate'}
LEVELS=('finish','find','none')

def select_assistance(history,index,threshold=.5,window=4):
 primary=history[-1].get('assistance_primary_level','finish') if history else 'finish'
 recent=[g for g in history if g.get('assistance_level')==primary and not g.get('assistance_retention')][-window:]
 if len(recent)>=window and primary!='none':
  scores=[v for g in recent for v in g.get('native_rewards',[]) if v is not None]
  if scores and sum(v==1 for v in scores)/len(scores)>=threshold:primary=LEVELS[LEVELS.index(primary)+1]
 replay=primary!='finish' and index%5==4
 level=LEVELS[max(0,LEVELS.index(primary)-1)] if replay else primary
 return dict(assistance_level=level,assistance_primary_level=primary,assistance_retention=replay)

def prepare_prefix(adapter,raw,level):
 if level=='none':return raw
 if level not in LEVELS:raise ValueError('Unknown assistance level')
 native=adapter.env._orchestrator.environment;db=native.tools.db;before=(native.get_db_hash(),native.get_user_db_hash());raw=copy.deepcopy(raw);events=[]
 def advance(text,call=None):
  if len(events)>=adapter.cfg.assisted_prefix_max_actions:raise RuntimeError('Assistance prefix action budget exceeded')
  if call and call['name'] not in READ_ONLY:raise RuntimeError('Assistance cannot execute mutations')
  incoming,done,_=adapter.step(text)
  message=dict(role='assistant',content=text)
  if call:
   call_id=next((m.get('tool_call_id') for m in incoming if m['role']=='tool'),None) or 'prefix-'+str(len(events))
   message['tool_calls']=[dict(id=call_id,type='function',function=copy.deepcopy(call))]
  raw.append(message);raw.extend(copy.deepcopy(incoming));events.append(dict(call=call,text=text,incoming=copy.deepcopy(incoming)))
  if done or any(failed(m) for m in incoming if m['role']=='tool'):raise RuntimeError('Assistance prefix failed or ended the task')
  return incoming
 def execute(name,args):
  c=dict(name=name,arguments=args);incoming=advance('<tool_call>'+json.dumps(c,separators=(',',':'))+'</tool_call>',c)
  response=next(m for m in incoming if m.get('name')==name);return tool_body(response)
 def customer_text():return '\n'.join(m['content'] for m in raw if m['role']=='user')
 # Obtain identity through the actual frozen customer, never invent a disclosure.
 email=None
 for attempt in range(2):
  emails=re.findall(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',customer_text());email=next((e for e in emails if any(u.email==e for u in db.users.values())),None)
  if email:break
  if attempt==0:advance('Could you please provide your email address so I can find your account?')
 if email:uid=execute('find_user_id_by_email',dict(email=email))
 else:
  advance('Could you please provide your first name, last name, and ZIP code so I can find your account?')
  text=customer_text();matches=[u for u in db.users.values() if all(observed(v,text) for v in [u.name.first_name,u.name.last_name,u.address.zip])]
  if len(matches)!=1:raise RuntimeError('Assistance could not obtain a unique observed identity')
  u=matches[0];uid=execute('find_user_id_by_name_zip',dict(first_name=u.name.first_name,last_name=u.name.last_name,zip=u.address.zip))
 if not isinstance(uid,str) or uid not in db.users:raise RuntimeError('Assistance did not authenticate a real user')
 if level=='finish':
  execute('get_user_details',dict(user_id=uid));task=adapter.info['task'];actions=task.evaluation_criteria.actions or []
  # The privileged selector chooses useful READS only. Reference mutation arguments,
  # approval, desired variant IDs and reference trajectories are never serialized.
  orders=sorted({a.arguments['order_id'] for a in actions if 'order_id' in a.arguments});products=set()
  for oid in orders:
   if oid not in db.orders or db.orders[oid].user_id!=uid:raise RuntimeError('Assistance target is outside authenticated ownership')
   result=execute('get_order_details',dict(order_id=oid));items={i for a in actions for i in a.arguments.get('item_ids',[])}
   products.update(i['product_id'] for i in result.get('items',[]) if i['item_id'] in items)
  products.update(a.arguments['product_id'] for a in actions if a.name=='get_product_details' and 'product_id' in a.arguments)
  for pid in sorted(products):execute('get_product_details',dict(product_id=pid))
 after=(native.get_db_hash(),native.get_user_db_hash())
 if before!=after:raise RuntimeError('Assistance prefix mutated database state')
 adapter.assistance=dict(level=level,prefix_actions=len(events),prefix_message_count=len(raw),read_calls=[e['call'] for e in events if e['call']],database_unchanged=True,policy_targets=False)
 return raw

def prime_tracker(tracker,raw):
 prior=[]
 for i,m in enumerate(raw):
  if m['role']=='assistant' and '<tool_call>' in m.get('content',''):
   incoming=[]
   for n in raw[i+1:]:
    if n['role']!='tool':break
    incoming.append(n)
   if incoming:
    # Schemas are provided by the live environment; caller attaches them.
    tracker.record(m['content'],prior,incoming,tracker.prefix_schemas)
  prior.append(m)
 tracker.events=[];tracker.streak=0;tracker.credited=set()
