"""Repeat the exact clean-data task with/without a direct clarification hint."""
import copy,json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_prompts import clean_tokenizer_template,check_system
from tau2_grpo.glaive_signals import score_decision


def main():
    torch.set_num_threads(8)
    uid='78a0f4fbf464e4b67629db47c8ee2cbfe77fba8dce55d420a8afbeea54294201:2:missing_required'
    row=next(json.loads(s) for s in Path('outputs/glaive-rl-signals-v2/train.jsonl').open() if json.loads(s)['id']==uid)
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-55f242da3a/checkpoint-2/policy',device='cpu',temperature=.7,top_p=1.)
    policy=LocalPolicy(cfg);clean_tokenizer_template(policy.tokenizer)
    out=Path('outputs/diagnostics/clean-task78-missing-text');out.mkdir(parents=True,exist_ok=True);results=[]
    for arm in ('fixed_system','fixed_system_direct_hint'):
        messages=copy.deepcopy(row['prompt']);check_system(messages[0]['content'],row['tools'])
        if arm=='fixed_system_direct_hint':messages[-1]['content']+='\n\nGuidance for your next response: Ask the customer for the missing text before calling the translation tool.\nCarry out this instruction now; do not discuss the guidance.'
        text=policy.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);assert text.count('<tools>')==text.count('</tools>')==1
        (out/(arm+'-rendered.txt')).write_text(text)
        ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        generated=policy.generate({'prompt_ids':ids},128,106);scores=score_decision(generated['text'],row)
        results.append({'arm':arm,'output':generated['text'],'clarified':bool(scores.get('clarification_correct')),'scores':scores,'messages':messages+[{'role':'assistant','content':generated['text']}],'ended_with_eos':generated['ended_with_eos']})
        (out/'results.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'device':'cpu','temperature':.7,'top_p':1.,'seed':106,'note':'Uses the policy snapshot at the original failed RL group; no training changes. CPU/GPU arithmetic may affect sampled outputs.','results':results},indent=2))
        (out/'conversations.txt').write_text('\n\n'.join('ARM: '+r['arm']+' | CLARIFIED: '+str(r['clarified'])+'\n\n'+'\n\n'.join(m['role'].upper()+': '+m['content'] for m in r['messages'][1:]) for r in results));print(json.dumps(results[-1]),flush=True)

if __name__=='__main__':main()
