"""Auditable additive Tau2 criteria, argument grounding, and failed-call streaks."""
import json,math,re
from jsonschema import validate,ValidationError
from tau2_grpo.prepare_function_calling import parse_calls

def component_sum(info):
 """Sum actual evaluated checks once; never count empty-category neutral ones."""
 rows=[]
 def add(category,criterion,value):
  value=float(value)
  if not math.isfinite(value):raise ValueError('Nonfinite reward component')
  rows.append({'category':category,'criterion':criterion,'value':value})
 if info.get('db_check') is not None:add('DB','final_database',info['db_check']['db_reward'])
 for i,c in enumerate(info.get('action_checks') or []):add('ACTION',c['action'].get('action_id',str(i)),c['action_reward'])
 for i,c in enumerate(info.get('env_assertions') or []):add('ENV_ASSERTION',str(i),c['reward'])
 for i,c in enumerate(info.get('nl_assertions') or []):add('NL_ASSERTION',c['nl_assertion'],int(c['met']))
 for i,c in enumerate(info.get('communicate_checks') or []):add('COMMUNICATE',c['info'],int(c['met']))
 return sum(r['value'] for r in rows),rows

def tool_body(message):
 content=message.get('content','')
 try:content=json.loads(content)
 except (ValueError,TypeError):pass
 if isinstance(content,dict) and 'name' in content and 'content' in content:content=content['content']
 return content

def failed(message):
 body=tool_body(message)
 return bool(message.get('error')) or message.get('name')=='__runtime__' or (isinstance(body,dict) and (bool(body.get('error')) or str(body.get('status','')).lower() in ('error','failed','failure'))) or (isinstance(body,str) and bool(re.match(r'^\s*(?:error\b|user not found\b|order not found\b)',body,re.I)))

def leaves(value):
 if isinstance(value,dict):
  for child in value.values():yield from leaves(child)
 elif isinstance(value,list):
  for child in value:yield from leaves(child)
 else:yield value

def observed(value,text):
 if isinstance(value,bool) or value is None:return True
 if isinstance(value,(int,float)):
  return any(float(n)==float(value) for n in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])',text))
 if not value:return True # Empty address2 is an allowed literal default.
 return re.search(r'(?<![\w@.+-])'+re.escape(str(value))+r'(?![\w@+-]|\.[\w])',text) is not None

def has_confirmation(prior):
 # One outstanding approval; later refusal or a consumed write invalidates it.
 asked=False;approved=False
 for m in prior:
  text=m.get('content','')
  if m.get('role')=='assistant':
   if re.search(r'"name"\s*:\s*"(?:cancel_|modify_|return_|exchange_)',text):asked=False;approved=False
   elif re.search(r'confirm|shall i|may i|would you like.*proceed|can i proceed',text,re.I):asked=True;approved=False
  elif m.get('role')=='user' and asked:
   if re.search(r"\b(?:no|do not|don't|stop|cancel that request)\b",text,re.I):asked=False;approved=False
   else:approved=bool(re.search(r'\b(?:yes|confirm|proceed|go ahead|approved|that is correct|sounds good)\b',text,re.I))
 return approved

class ToolRewardTracker:
 def __init__(self,cfg):
  self.cfg=cfg;self.events=[];self.streak=0;self.authenticated_id=None;self.orders={};self.credited=set()
 def record(self,text,prior,incoming,schemas):
  if '<tool_call>' not in text:return
  reasons=[];call=None
  try:
   calls=parse_calls(text)
   if len(calls)!=1:raise ValueError('Expected exactly one tool call')
   call=calls[0]
   if call['name'] not in schemas:raise ValueError('Unknown tool')
   schema=schemas[call['name']];validate(call['arguments'],schema)
   if set(call['arguments'])-set(schema.get('properties',{})):raise ValueError('Unknown argument keys')
  except (ValueError,ValidationError) as exc:reasons.append('schema/parse: '+str(exc)[:160])
  successful=bool(call) and bool([m for m in incoming if m.get('role')=='tool' and m.get('name')==call['name']]) and not any(failed(m) for m in incoming if m.get('role')=='tool')
  customer_sources='\n'.join(m.get('content','') for m in prior if m.get('role')=='user' and not m.get('content','').lstrip().startswith('<tool_response>'))
  sources='\n'.join(m.get('content','') for m in prior if (m.get('role')=='user' and not m.get('content','').lstrip().startswith('<tool_response>')) or (m.get('role')=='tool' and not failed(m)))
  if call and call['name'].startswith('find_user_id_'):sources=customer_sources
  if call and not reasons:
   for key,value in call['arguments'].items():
    if key=='reason' and value in ('no longer needed','ordered by mistake'):continue
    if key=='expression':
     if not re.fullmatch(r'[\d\s.+*/()\-]+',str(value)) or any(not observed(float(n),sources) for n in re.findall(r'\d+(?:\.\d+)?',str(value))):reasons.append('unsupported arithmetic operands')
    elif any(not observed(v,sources) for v in leaves(value)):reasons.append('unobserved argument: '+key)
   name=call['name'];args=call['arguments']
   if name.startswith(('cancel_','modify_','return_','exchange_')) and not has_confirmation(prior):reasons.append('no observed explicit confirmation before mutation')
   if name not in ('find_user_id_by_email','find_user_id_by_name_zip','calculate','done','transfer_to_human_agents'):
    if self.authenticated_id is None:reasons.append('no grounded successful authentication')
    if 'user_id' in args and args['user_id']!=self.authenticated_id:reasons.append('wrong authenticated user')
    if 'order_id' in args and name!='get_order_details':
     order=self.orders.get(args['order_id'])
     if not order or order.get('user_id')!=self.authenticated_id:reasons.append('order ownership not verified')
     if order:
      for key in ('item_ids','new_item_ids'):
       if key in args and key=='item_ids' and any(item not in [i['item_id'] for i in order.get('items',[])] for item in args[key]):reasons.append('items outside verified order')
   if name=='get_order_details' and successful:
    body=tool_body(next(m for m in incoming if m.get('name')==name))
    if not isinstance(body,dict) or body.get('user_id')!=self.authenticated_id:reasons.append('returned order belongs to another user')
  valid=call is not None and not reasons
  eligible=not (call and call['name'] in ('done','transfer_to_human_agents') and valid)
  credited=False
  if eligible:
   if successful and valid:self.streak=0
   else:self.streak+=1
   if valid and successful:
    key=json.dumps(call,sort_keys=True,ensure_ascii=False)
    if key not in self.credited:self.credited.add(key);credited=True
  failure_penalty=min(self.cfg.failed_call_penalty_cap,self.cfg.failed_call_penalty*max(0,self.streak-1)) if eligible and not (successful and valid) else 0.
  event={'turn':len(self.events),'call':call,'arguments_grounded':valid,'argument_errors':reasons,'successful':successful,'eligible':eligible,'credited_unique_correct_call':credited,'failure_streak':self.streak,'failure_penalty':failure_penalty}
  self.events.append(event)
  if valid and successful:
   body=tool_body(next(m for m in incoming if m.get('name')==call['name']))
   if call['name'] in ('find_user_id_by_email','find_user_id_by_name_zip') and isinstance(body,str):self.authenticated_id=body
   if call['name']=='get_order_details' and isinstance(body,dict) and body.get('user_id')==self.authenticated_id:self.orders[body['order_id']]=body
 def score(self):
  eligible=[e for e in self.events if e['eligible']];count=max(1,len(eligible))
  bonus=self.cfg.argument_reward_weight*sum(e['credited_unique_correct_call'] for e in eligible)/count
  incorrect=self.cfg.incorrect_argument_penalty*sum(not e['arguments_grounded'] for e in eligible)/count
  failure=sum(e['failure_penalty'] for e in eligible)
  return {'argument_bonus':bonus,'incorrect_argument_penalty':incorrect,'consecutive_failure_penalty':failure,'tool_attempts':len(eligible),'unique_correct_calls':sum(e['credited_unique_correct_call'] for e in eligible),'max_failure_streak':max((e['failure_streak'] for e in eligible),default=0)}

def shaped_reward(native,simulation,tracker):
 info=simulation['reward_info'];excluded=[]
 if tracker.cfg.assistance_level!='none':
  import copy
  info=copy.deepcopy(info)
  for check in info.get('action_checks') or []:
   a=check['action'];keys=a.get('compare_args');name=a['name'];wanted=a.get('arguments',{})
   matched=any(e.get('call') and e['call']['name']==name and (e['call']['arguments']==wanted if keys is None else all(e['call']['arguments'].get(k)==wanted.get(k) for k in keys)) for e in tracker.events)
   if not matched and check.get('action_reward',0):excluded.append(a.get('action_id',name));check['action_reward']=0.
 total,components=component_sum(info);parts=tracker.score()
 reward=total+parts['argument_bonus']-parts['incorrect_argument_penalty']-parts['consecutive_failure_penalty']
 if not math.isfinite(reward):raise ValueError('Nonfinite shaped reward')
 return reward,{'version':'tau2-additive-penalties-v2' if tracker.cfg.argument_reward_weight==0 else 'tau2-additive-grounded-v1','grounded_call_bonus_enabled':tracker.cfg.argument_reward_weight!=0,'native_reward':native,'excluded_unsampled_action_credits':excluded,'tau2_sum':total,'tau2_components':components,**parts,'total':reward,'note':'Grounding/schema checks are conservative heuristics, not a semantic correctness oracle. Task-specific correctness is also scored by Tau2 action checks.'}
