"""Sample16 suffix rollouts per selected state before spending on updates."""
import argparse,json
from pathlib import Path
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_rollout import atomic_json

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();cfg=load_settings(a.config);assert cfg.group_size==16 and cfg.bank_view=='midturn_mix'
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_transformers_workers import collect_group
 policy=LocalPolicy(cfg,trainable=False);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 cases=[('midturn','after-reply::world-0000-missing_zip'),('midturn','after-reply::world-0001-partial_last_name'),('midturn','after-reply::world-0002-missing_first_name'),('opening','world-0000-missing_zip'),('retention','world-0000-email_lookup')];allowed=set(task_ids(cfg));results=[]
 for index,(kind,cid) in enumerate(cases):
  assert cid in allowed
  directory=out/f'group-{index:04d}';episodes=collect_group(policy,cfg,cid,cfg.seed+index*cfg.group_size,directory,with_reference=False)
  if not all(e['complete'] for e in episodes):raise RuntimeError('Midturn smoke has infrastructure errors')
  successes=sum(e['short_task_success'] for e in episodes);row=dict(category=kind,task_id=cid,rollouts=len(episodes),successes=successes,mixed_rewards=0<successes<len(episodes),rewards=[e['reward'] for e in episodes]);results.append(row)
  summary=dict(groups=results,targeted_midturn_mixed_groups=sum(r['mixed_rewards'] for r in results if r['category']=='midturn'),training_updates=0,native_tau2_evaluation=False)
  atomic_json(out/'summary.json',summary);print(json.dumps(dict(event='midturn_smoke_group',**row)),flush=True)
 print('MIDTURN_SMOKE_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
