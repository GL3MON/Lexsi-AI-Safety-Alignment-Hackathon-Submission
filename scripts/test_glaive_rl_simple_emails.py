"""Adversarial exact-copy and missing-slot probes; no API execution."""
import json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.prepare_function_calling import parse_calls


def main():
    torch.set_num_threads(8)
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy',temperature=.7,top_p=.95)
    policy=LocalPolicy(cfg)
    row=next(json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open() if json.loads(s)['scenario']=='missing_required' and json.loads(s)['reward_reference']['missing_field']=='email')
    system=row['prompt'][0]
    old='mira7@example.org';correct='mira07@example.org'
    cases=[{'id':'corrected_email_single_zero_difference','expected':{'name':'generate_password_reset_token','arguments':{'email':correct}},'messages':[{'role':'user','content':f'I need a password reset. My email is {old}.'},{'role':'assistant','content':'<tool_call>'+json.dumps({'name':'generate_password_reset_token','arguments':{'email':old}})+'</tool_call>'},{'role':'user','content':'<tool_response>'+json.dumps({'name':'generate_password_reset_token','content':{'error':'Account not found'}})+'</tool_response>'},{'role':'assistant','content':'Please check the email address.'},{'role':'user','content':f'I missed one zero. Use exactly {correct}. The old address is wrong. Preserve the spelling and both digits.'}]},
    {'id':'incomplete_email_must_ask','expected':None,'messages':[{'role':'user','content':'I need a password reset. My email starts with mira but I cannot remember the part after @. Do not guess the domain.'}]},
    {'id':'two_addresses_latest_authoritative','expected':{'name':'generate_password_reset_token','arguments':{'email':'keira@example.net'}},'messages':[{'role':'user','content':'I need a password reset. My colleague uses tavish@example.org. My old address was keira@example.org. My CURRENT account email is keira@example.net. Use my current email only.'}]}]
    out=Path('outputs/diagnostics/glaive-rl16-simple-email-cases');out.mkdir(parents=True,exist_ok=True);results=[]
    for case in cases:
        for seed in (42,43,44,45):
            ids=policy.tokenizer.apply_chat_template([system]+case['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
            generated=policy.generate({'prompt_ids':ids},256,seed)
            try:calls=parse_calls(generated['text'])
            except ValueError:calls=None
            passed=calls==[case['expected']] if case['expected'] else calls==[] and '<tool_call' not in generated['text'] and any(t in generated['text'].lower() for t in ('email','domain','address')) and any(t in generated['text'].lower() for t in ('?','please provide','please share','could you','need'))
            results.append({'case':case['id'],'seed':seed,'passed':passed,'expected':case['expected'],'output':generated['text'],'messages':case['messages']+[{'role':'assistant','content':generated['text']}]})
            (out/'results.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'temperature':.7,'top_p':.95,'results':results},indent=2))
            (out/'conversations.txt').write_text('\n\n'.join('CASE: '+r['case']+' | SEED: '+str(r['seed'])+' | PASS: '+str(r['passed'])+'\nEXPECTED: '+json.dumps(r['expected'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages']) for r in results))
            print(case['id'],seed,passed,repr(generated['text']),flush=True)

if __name__=='__main__':main()
