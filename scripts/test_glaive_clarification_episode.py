"""One live two-decision diagnostic, retaining the actual assistant response."""
import json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_signals import score_decision


def main():
    torch.set_num_threads(8)
    rows=[json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()]
    row=next(r for r in rows if r['scenario']=='missing_required' and r['reward_reference']['missing_field']=='email')
    follow=next(r for r in rows if r['source_id']==row['source_id'] and r['scenario']=='clarification_followup')
    cfg=RetailRLConfig(initial_adapter='outputs/sft-glaive/run-d9133530a1/checkpoint-475',temperature=.7,top_p=.95)
    policy=LocalPolicy(cfg)
    messages=list(row['prompt']);results=[]
    out=Path('outputs/diagnostics/glaive-clarification-episode');out.mkdir(parents=True,exist_ok=True)
    for i,reference in enumerate((row,follow)):
        if i:messages.append(follow['prompt'][-1])
        ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        result=policy.generate({'prompt_ids':ids},256,42+i)
        details=score_decision(result['text'],reference)
        messages.append({'role':'assistant','content':result['text']})
        results.append({'turn':i+1,'output':result['text'],'reward_details':details,'ended_with_eos':result['ended_with_eos']})
        (out/'episode.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'temperature':.7,'top_p':.95,'messages':messages,'turns':results,'note':'Customer supplies missing email on turn two even if first response fails; this is a controlled diagnostic, not execution of a password-reset API.'},indent=2))
        (out/'conversation.txt').write_text('CHECKPOINT: '+cfg.initial_adapter+'\nTEMPERATURE: 0.7 | TOP_P: 0.95\n\n'+'\n\n'.join(m['role'].upper()+': '+m['content'] for m in messages[1:])+'\n\nSCORES:\n'+json.dumps(results,indent=2))
        print(json.dumps(results[-1]),flush=True)

if __name__=='__main__':main()
