"""Paired greedy evaluation on the fixed retail validation bank; no Tau2 or training."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
from tau2_grpo.rl_config import load_settings,path
from tau2_grpo.rl_bank import validation_cases
from tau2_grpo.rl_bank_eval import verify_episode_prompt
from tau2_grpo.rl_rollout import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--baseline',required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 from tau2_grpo.rl import make_episode
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_reporting import write_episode_conversation
 import wandb
 cfg=replace(load_settings(a.config),temperature=0.);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 local=LocalPolicy(cfg,trainable=False);local.model.load_adapter(a.baseline,adapter_name='baseline',is_trainable=False)
 class Policy:
  def __init__(self,alias):self.alias=alias;self.tokenizer=local.tokenizer;self.version=0
  def generate(self,*args,**kwargs):local.model.set_adapter(self.alias);return local.generate(*args,**kwargs)
 results={};ids=validation_cases(cfg)
 meta={'inference_only':True,'native_tau2':False,'temperature':0.,'validation_count':len(ids),'baseline_adapter':a.baseline,'candidate_adapter':cfg.initial_adapter,'candidate_sha256':hashlib.sha256((path(cfg.initial_adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()}
 with wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name='assisted-rl-checkpoint2-heldout-greedy',config=meta) as wb:
  atomic_json(out/'metadata.json',meta)
  for stage,alias in [('short-rl-checkpoint-7','baseline'),('assisted-rl-checkpoint-2','policy')]:
   policy=Policy(alias);rows=[];seen=set();d=out/stage;d.mkdir(parents=True,exist_ok=True)
   for cid in ids:
    ep=make_episode(policy,cfg,cid,cfg.seed,d/(cid+'-bank.json'))
    if not ep['complete']:raise RuntimeError('Incomplete evaluation episode '+cid+': '+str(ep.get('error')))
    verify_episode_prompt(ep,seen);write_episode_conversation(d/(cid+'-bank.json'),ep)
    row={'task_id':cid,'success':bool(ep.get('short_task_success')),'turns':len(ep['actions']),'complete':ep['complete']};rows.append(row)
    atomic_json(d/'summary.json',{'results':rows,'passed':sum(x['success'] for x in rows),'total':len(ids),'native_tau2_evaluation':False})
    wb.log({stage+'/case_success':int(row['success']),'case_index':len(rows)})
   results[stage]={'passed':sum(x['success'] for x in rows),'total':len(rows)}
  atomic_json(out/'summary.json',results);wb.log({'comparison/'+k:v['passed']/v['total'] for k,v in results.items()})
  (out/'COMPLETE').write_text('complete\\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
