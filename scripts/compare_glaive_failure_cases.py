"""Matched diagnostics on early low-reward training cases (not held-out eval)."""
import gc,json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows,rollout


def main():
    torch.set_num_threads(8)
    run=Path('outputs/rl-glaive/run-e864eb346e')
    groups=json.load((run/'groups.json').open())
    chosen=[g for g in groups if g['mean_reward']<.8][:3]
    rows={r['id']:r for r in episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()])}
    cases=[]
    for group in chosen:
        original=json.load((run/f"group-{group['group']:04d}"/'episode-00.json').open())
        cases.append((group['group'],rows[original['task_id']]))
    out=Path('outputs/diagnostics/glaive-rl16-failure-case-comparison');out.mkdir(parents=True,exist_ok=True)
    results=[]
    def save():
        (out/'results.json').write_text(json.dumps({'selection':'First three training groups with mean reward < 0.8; seen-case diagnostic, not held-out performance.','results':results},indent=2))
        summary=[]
        for index,row in cases:
            summary.append({'training_group':index,'scenario':row['scenario'],'task_id':row['id'],**{label:{'correct':sum(r['passed'] for r in results if r['model']==label and r['training_group']==index),'total':sum(r['model']==label and r['training_group']==index for r in results),'mean_reward':sum(r['episode']['reward'] for r in results if r['model']==label and r['training_group']==index)/max(1,sum(r['model']==label and r['training_group']==index for r in results))} for label in ('sft-475','rl-16')}})
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'conversations.txt').write_text('Seen training cases; diagnostic only.\n\n'+'\n\n'.join('MODEL: '+r['model']+' | TRAINING GROUP: '+str(r['training_group'])+' | SEED: '+str(r['seed'])+' | REWARD: '+str(r['episode']['reward'])+' | PASS: '+str(r['passed'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['episode']['messages'][1:]) for r in results))
    for label,adapter in [('sft-475','outputs/sft-glaive/run-d9133530a1/checkpoint-475'),('rl-16','outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy')]:
        cfg=RetailRLConfig(initial_adapter=adapter,temperature=.7,top_p=1.,max_actions=10,response_tokens=256)
        policy=LocalPolicy(cfg)
        for group,row in cases:
            for i in range(4):
                seed=42+group*8+i;e=rollout(policy,cfg,row,seed,str(i))
                results.append({'model':label,'training_group':group,'seed':seed,'passed':e['termination_reason']=='correct_call','episode':e});save();print(label,group,seed,e['reward'],e['termination_reason'],flush=True)
        del policy;gc.collect();torch.cuda.empty_cache()
    (out/'DONE').write_text('Complete\n')

if __name__=='__main__':main()
