"""Batched medium workflow smoke; verified-progress variance gates training."""
import argparse,json
from pathlib import Path
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_rollout import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();c=load_settings(a.config);assert c.bank_view=='medium_mix' and c.group_size==16
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_transformers_workers import collect_group
 policy=LocalPolicy(c,trainable=False);out=Path(a.output);results=[]
 for i,family in enumerate(['status_missing_zip','cancel_email','exchange_item']):
  cid=f'world-{i:04d}-medium-{family}';episodes=collect_group(policy,c,cid,c.seed+16*i,out/f'group-{i:04d}',with_reference=False)
  if not all(e['complete'] for e in episodes):raise RuntimeError('Medium smoke infrastructure error')
  row=dict(task_id=cid,successes=sum(e['short_task_success'] for e in episodes),rewards=[e['reward'] for e in episodes],actor_turns=[len(e['actions']) for e in episodes],mixed_rewards=len({e['reward'] for e in episodes})>1);results.append(row)
  summary=dict(results=results,mixed_groups=sum(r['mixed_rewards'] for r in results),training_updates=0);atomic_json(out/'summary.json',summary);print(json.dumps(dict(event='medium_smoke_group',**row)),flush=True)
if __name__=='__main__':main()
