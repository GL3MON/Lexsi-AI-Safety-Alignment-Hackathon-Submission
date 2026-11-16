"""Native environment smoke: null infrastructure rewards block subsequent training."""
import argparse,json,math
from pathlib import Path
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_rollout import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();cfg=load_settings(a.config)
 from tau2_grpo.tau2_curriculum import curriculum_order
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_transformers_workers import collect_group
 cid=curriculum_order(cfg,task_ids(cfg))[0];policy=LocalPolicy(cfg,trainable=False);out=Path(a.output);cached={};cache_verified=(out/'CACHE_VERIFIED').is_file()
 if cache_verified:
  for index in range(cfg.group_size):
   file=out/'group-0000'/f'episode-{index:02d}.json'
   if file.exists():
    e=json.loads(file.read_text())
    if e.get('complete') and e.get('reward') is not None and e.get('task_id')==cid and e.get('seed')==cfg.seed+index and e.get('policy_version')==policy.version:cached[index]=e
 missing=[i for i in range(cfg.group_size) if i not in cached]
 print(json.dumps(dict(event='native_smoke_cache',reused=len(cached),collecting=len(missing))),flush=True)
 fresh=collect_group(policy,cfg,cid,cfg.seed,out/'group-0000',with_reference=False,rollout_indices=missing) if missing else []
 cached.update(zip(missing,fresh,strict=True));episodes=[cached[i] for i in range(cfg.group_size)]
 rows=[dict(task_id=e['task_id'],native_reward=e.get('native_reward'),reward=e['reward'],complete=e['complete'],turns=len(e['actions']),error=e.get('error')) for e in episodes];atomic_json(out/'summary.json',dict(results=rows,task_id=cid,rollouts=len(rows),native_successes=sum(r['native_reward']==1 for r in rows),infrastructure_errors=sum(not r['complete'] for r in rows),reward_variance=len({r['reward'] for r in rows})>1))
 if any(not e['complete'] or e['reward'] is None or not math.isfinite(e['reward']) for e in episodes):raise RuntimeError('Native Tau2 smoke has infrastructure failures; training blocked')
 print('NATIVE_TAU2_SMOKE_COMPLETE',json.dumps(rows),flush=True)
if __name__=='__main__':main()
