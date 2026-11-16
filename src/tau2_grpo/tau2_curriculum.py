"""Train-only task complexity proxy; no hidden task answers enter actor prompts."""
import json,math,random
from pathlib import Path
READS={'find_user_id_by_email','find_user_id_by_name_zip','get_user_details','get_order_details','get_product_details','list_all_product_types','calculate'}
def make_curriculum(tasks,manifest,groups=30,seed=345):
 allowed=set(manifest['partitions']['rl']);by_id={str(t['id']):t for t in tasks};rows=[]
 for tid in allowed:
  task=by_id[tid];evaluation=task.get('evaluation_criteria') or {};actions=evaluation.get('actions') or [];writes=sum(a['name'] not in READS for a in actions);nl=len(evaluation.get('nl_assertions') or []);env=len(evaluation.get('env_assertions') or []);orders={a.get('arguments',{}).get('order_id') for a in actions};orders.discard(None)
  score=len(actions)+2*writes+nl+env+max(0,len(orders)-1)
  rows.append(dict(task_id=tid,complexity=score,reference_actions=len(actions),write_actions=writes,nl_checks=nl,env_checks=env,distinct_orders=len(orders)))
 rows.sort(key=lambda r:(r['complexity'],int(r['task_id'])));deferred=[r for r in rows if r['reference_actions']>8];rows=[r for r in rows if r['reference_actions']<=8];width=math.ceil(len(rows)/3);bins=[rows[:width],rows[width:2*width],rows[2*width:]];rng=random.Random(seed);schedule=[]
 for stage,pool in enumerate(bins):
  cycle=[];prior=[r for earlier in bins[:stage] for r in earlier];n=groups//3+(stage<groups%3)
  for i in range(n):
   retention=bool(prior) and i%5==4
   if retention:row=rng.choice(prior)
   else:
    if not cycle:
     cycle=pool.copy()
     if stage==0:cycle.reverse()
     else:rng.shuffle(cycle)
    row=cycle.pop()
   schedule.append(dict(group=len(schedule),stage=['easy','medium','hard'][stage],retention=retention,**row))
 return dict(version='tau2-complexity-curriculum-v1',source='Original reserved RL task IDs only',difficulty_note='Static reference-action/check-count proxy, not measured task difficulty',ranked_tasks=rows,deferred_tasks=deferred,deferral_reason='Reference requires more than 8 tool actions; defer under current 10-actor-action cap',schedule=schedule)

def curriculum_order(cfg,ids):
 from tau2_grpo.rl_config import path
 data=json.loads(path(cfg.curriculum_file).read_text());order=[r['task_id'] for r in data['schedule']]
 if len(order)<cfg.train_steps or not set(order)<=set(ids):raise ValueError('Curriculum outside allowed RL split or too short')
 return order[:cfg.train_steps]
