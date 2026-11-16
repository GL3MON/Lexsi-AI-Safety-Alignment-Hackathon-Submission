#!/usr/bin/env python3
"""Full official retail test split, paired across the three completed SFT stages."""
import argparse,csv,gc,hashlib,json,os,random,time
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path
from tau2_grpo.rl_config import RetailRLConfig,ROOT,task_ids
from tau2_grpo.rl_rollout import atomic_json
from tau2_grpo.rl_reporting import write_episode_conversation

class InferenceOnlyPolicy:
 """Skip training-only probability replay without changing decoding or scoring."""
 def __init__(self,policy):self.policy=policy;self.tokenizer=policy.tokenizer;self.version=0
 def generate(self,*args,**kwargs):return self.policy.generate(*args,**kwargs)

def main(args):
 import torch,wandb
 from loguru import logger
 from openai import OpenAI
 from tau2_grpo.rl import make_episode
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2.domains.retail.environment import get_tasks_split
 logger.remove();logger.add(__import__('sys').stderr,level='ERROR');torch.set_num_threads(8)
 os.chdir(ROOT);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 cfg=RetailRLConfig(temperature=.7,top_p=.95,group_size=1,agent_prompt_file='templates/retail_eval_tool_preamble.txt')
 allowed=task_ids(cfg,'test');assert set(allowed)==set(get_tasks_split()['test'])
 ids=sorted(allowed,key=int);random.Random(42).shuffle(ids)
 if args.limit:ids=ids[:args.limit]
 source=json.loads((ROOT/'outputs/diagnostics/completed-sft-stages-four-scenarios/manifest.json').read_text());stages=source['stages']
 client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=20);models=client.models.list().model_dump();client.close()
 assert cfg.user_model in {m['id'] for m in models['data']}
 config={'settings':asdict(cfg),'task_ids':ids,'stages':stages,'episodes_planned':len(ids)*len(stages),'task_split':'official Tau2 retail test','split_manifest_sha256':hashlib.sha256((ROOT/'outputs/retail-data/split_manifest.json').read_bytes()).hexdigest(),'notes':['One sampled episode per task/model; not pass@4.','No training or parameter updates.','Qwen3.5-4B is the frozen customer, summarizer and NL assertion judge; this differs from benchmarks using another judge.','Tool-call preamble text is not delivered to customer by the current Tau2 adapter.','40 actor actions / 16384 generated-token budget; 8192 actor context with frozen summarizer.']}
 cp=out/'config.json'
 if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Evaluation resume config mismatch')
 atomic_json(cp,config)
 wb=wandb.init(project='tau2-retail',mode='online',name=out.name,dir=str(out),job_type='heldout-test-evaluation',config=config)
 atomic_json(out/'runtime.json',{'wandb_url':wb.url,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'endpoint_models':models})
 rows=[]
 def summarize(status):
  summaries=[]
  for stage in stages:
   selected=[r for r in rows if r['stage']==stage['stage']];scored=[r for r in selected if r['complete'] and r['reward'] is not None]
   checks=Counter((check['category'],check['status']) for r in selected for check in r['checks'])
   summaries.append({'stage':stage['stage'],'checkpoint':stage['checkpoint'],'episodes_finished':len(selected),'planned_tasks':len(ids),'scored_episodes':len(scored),'infrastructure_failures':sum(not r['complete'] for r in selected),'successful_tasks':sum(r['reward']==1 for r in scored),'success_rate_on_scored':sum(r['reward']==1 for r in scored)/len(scored) if scored else None,'mean_reward':sum(r['reward'] for r in scored)/len(scored) if scored else None,'mean_actor_turns':sum(r['actor_turns'] for r in selected)/len(selected) if selected else None,'check_counts':{cat:{s:checks[(cat,s)] for s in ('pass','fail','not_evaluated')} for cat in sorted({k[0] for k in checks})}})
  atomic_json(out/'summary.json',summaries);atomic_json(out/'episodes.json',rows);atomic_json(out/'progress.json',{'status':status,'finished_episodes':len(rows),'planned_episodes':len(ids)*len(stages),'updated_at':time.time()})
  text=['Tau2 retail official test split comparison','Settings: temperature 0.7, top-p 0.95, seed 42; one attempt per task/model.','STAGE | FINISHED | SUCCESS / SCORED | INFRA FAILURES | MEAN REWARD']
  text += [f"{s['stage']} | {s['episodes_finished']}/{len(ids)} | {s['successful_tasks']}/{s['scored_episodes']} | {s['infrastructure_failures']} | {s['mean_reward']}" for s in summaries]
  text += ['','Per-task transcripts: <stage>/task-<id>.conversation.txt','NL checks use local Qwen3.5-4B; these are not official leaderboard scores.']
  (out/'summary.txt').write_text('\n'.join(text))
  with (out/'scores.csv').open('w') as file:
   writer=csv.DictWriter(file,fieldnames=['stage','task_id','reward','complete','actor_turns','tool_calls','stop_reason','error','transcript']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
 success=False
 try:
  # Load all three small adapters on one base; switch them without reloading weights per task.
  local=LocalPolicy(replace(cfg,initial_adapter=stages[0]['checkpoint']))
  aliases={stages[0]['stage']:'policy'}
  for i,stage in enumerate(stages[1:],1):
   alias='eval_stage_'+str(i);local.model.load_adapter(stage['checkpoint'],adapter_name=alias,is_trainable=False);aliases[stage['stage']]=alias
  proxy=InferenceOnlyPolicy(local);consecutive_errors=0
  for task in ids:
   for stage in stages:
    label=stage['stage'];file=out/label/f'task-{task}.json'
    if file.exists() and json.loads(file.read_text()).get('check_report'):
     episode=json.loads(file.read_text())
    else:
     local.model.set_adapter(aliases[label]);local.model.requires_grad_(False);local.model.eval()
     summarize('running '+label+' task '+task)
     episode=make_episode(proxy,replace(cfg,initial_adapter=stage['checkpoint']),task,42,file)
    transcript=write_episode_conversation(file,episode);report=episode['check_report']
    row={'stage':label,'task_id':task,'reward':episode['reward'],'complete':episode['complete'],'actor_turns':len(episode['actions']),'tool_calls':sum(len(m.get('tool_calls') or []) for m in episode['messages'] if m['role']=='assistant'),'stop_reason':episode.get('stop_reason'),'error':episode.get('error'),'checks':report['checks'],'turn_counts':report['turn_counts'],'transcript':str(transcript)}
    rows.append(row);summarize('running');print('EPISODE DONE',json.dumps({k:v for k,v in row.items() if k not in ('checks','turn_counts')}),flush=True)
    if row['complete']:
     consecutive_errors=0;wb.log({label+'/reward':row['reward'],label+'/actor_turns':row['actor_turns'],'finished_episodes':len(rows)})
    else:consecutive_errors+=1
    if consecutive_errors>=3:raise RuntimeError('Three consecutive infrastructure failures; stopped rather than reporting them as model failures')
  summarize('complete');success=True;print('DONE',out,flush=True)
 finally:
  wb.finish(exit_code=0 if success else 1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/retail-eval/official-test-three-sft-stages');p.add_argument('--limit',type=int,default=0);a=p.parse_args();main(a)
