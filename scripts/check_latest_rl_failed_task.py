"""One unhinted connected episode from a selected latest clean-RL snapshot."""
import argparse,json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows,rollout
from tau2_grpo.glaive_prompts import clean_tokenizer_template


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    torch.set_num_threads(8)
    task='78a0f4fbf464e4b67629db47c8ee2cbfe77fba8dce55d420a8afbeea54294201:2:missing_required'
    row=next(r for r in episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals-v2/train.jsonl').open()]) if r['id']==task)
    cfg=RetailRLConfig(initial_adapter=a.checkpoint,device='cpu',temperature=.7,top_p=1.,response_tokens=256,max_actions=10)
    policy=LocalPolicy(cfg);clean_tokenizer_template(policy.tokenizer)
    e=rollout(policy,cfg,row,106,'106')
    name=Path(a.checkpoint).parent.name
    out=Path('outputs/diagnostics')/('clean-rl-'+name+'-failed-task78-unhinted');out.mkdir(parents=True,exist_ok=True)
    (out/'episode.json').write_text(json.dumps({'checkpoint':a.checkpoint,'device':'cpu','temperature':.7,'top_p':1.,'seed':106,'hints':False,'episode':e},indent=2,ensure_ascii=False))
    (out/'conversation.txt').write_text('CHECKPOINT: '+a.checkpoint+'\nNO HINTS | seed=106 | temperature=0.7 | top_p=1.0\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in e['messages'][1:])+'\n\nREWARD: '+str(e['reward'])+'\nTERMINATION: '+e['termination_reason'])
    print(json.dumps({'output_dir':str(out),'reward':e['reward'],'clarified':e['customer_replied'],'termination':e['termination_reason'],'outputs':[x['text'] for x in e['actions']]}),flush=True)

if __name__=='__main__':main()
