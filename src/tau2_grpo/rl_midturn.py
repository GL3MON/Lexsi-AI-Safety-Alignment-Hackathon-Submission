"""Verified mid-conversation initial states; no policy loss on supplied prefixes."""
import copy,json
from collections import defaultdict
from tau2_grpo.short_scenario_bank import BankEnvironment
MID_PREFIX='after-reply::'

def derive_case(case,world,catalog):
 original=copy.deepcopy(case);route=copy.deepcopy(original['routes'][int(original['world_id'].split('-')[1])%len(original['routes'])])
 if route[0]['kind']!='ask' or not route[0].get('reply') or len(route)<2:raise ValueError('Case has no clarification-to-reply transition')
 step=route[0];label={'email':'email address','first_name':'first name','last_name':'last name','zip':'ZIP code'}[step['field']]
 question='Could you please provide your '+('correct ' if step.get('correction') else '')+label+'?'
 proof=copy.deepcopy(original);proof['routes']=[route];env=BankEnvironment(proof,world,catalog);messages=env.reset();incoming,done,_=env.step(question)
 if done or not env.checks[-1]['passed'] or incoming!=[dict(role='user',content=step['reply'])]:raise ValueError('Invalid clarification/reply prefix')
 messages.append(dict(role='assistant',content=question));messages.extend(copy.deepcopy(incoming));env.close()
 original.update(id=MID_PREFIX+case['id'],messages=messages,routes=[route[1:]],max_actor_turns=max(len(route)-1,1),source_case_id=case['id'],curriculum_start='after_customer_reply',prefix_note='Verified clarification and customer reply are fixed context, never sampled actions or policy targets.')
 return original

def mixed_order(cfg,ids,rng):
 from tau2_grpo.rl_bank import load_bank,CLARIFICATION_FAMILIES
 _,rows,_,_=load_bank(cfg.scenario_bank);groups=defaultdict(lambda:defaultdict(list))
 for cid in ids:
  base_id=cid.removeprefix(MID_PREFIX);family=rows[base_id]['family']
  category='midturn' if cid.startswith(MID_PREFIX) else 'opening' if family in CLARIFICATION_FAMILIES else 'retention'
  groups[category][family].append(cid)
 if set(groups)!={'midturn','opening','retention'}:raise ValueError('Midturn mix requires all three categories')
 family_cycles={c:[] for c in groups};case_cycles={};pattern=('midturn','midturn','opening','midturn','retention');result=[]
 for i in range(len(ids)):
  cat=pattern[i%5]
  if not family_cycles[cat]:family_cycles[cat]=sorted(groups[cat]);rng.shuffle(family_cycles[cat])
  family=family_cycles[cat].pop();key=(cat,family)
  if not case_cycles.get(key):case_cycles[key]=list(groups[cat][family]);rng.shuffle(case_cycles[key])
  result.append(case_cycles[key].pop())
 return result
