"""Inference-only evaluation on the fixed 30-family bank validation subset."""
import argparse,json,hashlib,os
from dataclasses import asdict
from pathlib import Path
from tau2_grpo.rl_config import load_settings,path
from tau2_grpo.rl_bank import validation_cases
from tau2_grpo.rl_rollout import atomic_json

def verify_episode_prompt(episode,seen):
 from tau2_grpo.rl_context import student_messages
 first=episode['actions'][0];signature=tuple(first['prompt_ids'])
 if signature in seen:raise ValueError('Distinct retail cases reused a rendered prompt')
 if first['messages']!=student_messages(episode['messages'][:first['raw_message_count']]):raise ValueError('Rendered prompt differs from this episode history')
 seen.add(signature)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--greedy-baseline',required=True);parser.add_argument('--checkpoint-name',default='28');parser.add_argument('--worlds',default='');args=parser.parse_args()
 cfg=load_settings(args.config);assert cfg.environment=='short_bank'
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 baseline=json.loads(Path(args.greedy_baseline).read_text());before={r['task_id']:r for r in baseline['results']};ids=validation_cases(cfg)
 if args.worlds:
  from tau2_grpo.rl_bank import load_bank
  _,rows,pools,_=load_bank(cfg.scenario_bank);worlds={f'world-{int(w):04d}' for w in args.worlds.split(',')}
  ids=sorted(cid for cid in pools['validation'] if rows[cid]['world_id'] in worlds)
  assert len(ids)==30*len(worlds),'Unknown or non-validation world'
  assert set(before)<=set(ids),'Requested worlds exclude baseline cases'
 else:assert set(ids)==set(before),'Evaluation case set differs from original greedy validation'
 from tau2_grpo.rl import make_episode
 from tau2_grpo.rl_policy import LocalPolicy
 import wandb
 local=LocalPolicy(cfg,trainable=False)
 # Inference needs generated probabilities, but no GRPO replay or reference scoring.
 class InferencePolicy:
  tokenizer=local.tokenizer
  version=int(args.checkpoint_name) if args.checkpoint_name.isdigit() else 0
  def generate(self,*args,**kwargs):return local.generate(*args,**kwargs)
 policy=InferencePolicy();results=[];seen_prompts=set()
 metadata=dict(checkpoint=args.checkpoint_name,adapter_sha256=hashlib.sha256((path(cfg.initial_adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),config=asdict(cfg),generation_config=dict(temperature=cfg.temperature,top_p=cfg.top_p,top_k=0,repetition_penalty=1.0,response_tokens=cfg.response_tokens),native_tau2_evaluation=False,training=False,note='One sampled attempt per fixed validation case; not pass@k or an estimate over seeds.')
 atomic_json(output/'metadata.json',metadata)
 with wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name='short-bank-'+args.checkpoint_name+'-temp'+str(cfg.temperature),config=metadata) as wb:
  atomic_json(output/'runtime.json',dict(wandb_url=wb.url))
  for cid in ids:
   episode=make_episode(policy,cfg,cid,cfg.seed,output/(cid+'-bank.json'))
   if episode['complete']:verify_episode_prompt(episode,seen_prompts)
   row=dict(task_id=cid,scenario='bank',success=episode.get('short_task_success'),complete=episode['complete'],turns=len(episode['actions']),greedy_success=before.get(cid,{}).get('success'),stop_reason=episode.get('stop_reason'),error=episode.get('error'))
   results.append(row)
   summary=dict(results=results,passed=sum(bool(r['success']) for r in results),total=len(results),target=len(ids),complete=len(results)==len(ids),infrastructure_errors=sum(not r['complete'] for r in results),greedy_passed=baseline['passed'],greedy_total=baseline['total'],native_tau2_evaluation=False,temperature=cfg.temperature,checkpoint=args.checkpoint_name)
   atomic_json(output/'summary.json',summary)
   wb.log(dict(case_index=len(results),success=int(bool(row['success'])),turns=row['turns']))
   print(json.dumps(dict(event='sampled_validation_case',**row)),flush=True)
  wb.log({'eval/success_rate':summary['passed']/len(ids),'eval/samples':wandb.Table(columns=['task_id','success','greedy_success','turns','complete'],data=[[r[k] for k in ['task_id','success','greedy_success','turns','complete']] for r in results])})
  print('EVAL_COMPLETE',json.dumps(summary),flush=True)
  if summary['infrastructure_errors']:raise RuntimeError('Evaluation has infrastructure errors; do not treat them as failures')
if __name__=='__main__':main()
