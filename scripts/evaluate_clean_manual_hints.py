"""Fixed 20-case manual-hint baseline with exact rendered prompt exports."""
import copy,hashlib,json
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_prompts import check_system,clean_tokenizer_template
from tau2_grpo.glaive_signals import score_decision


def main():
    torch.set_num_threads(8);out=Path('outputs/diagnostics/clean-manual-hint-baseline')
    source=(out/'cases.jsonl').read_bytes();cases=[json.loads(s) for s in source.decode().splitlines()]
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy',temperature=.7,top_p=1.,response_tokens=256)
    policy=LocalPolicy(cfg);clean_tokenizer_template(policy.tokenizer);results=[];rendered=out/'rendered';rendered.mkdir(exist_ok=True)
    (out/'template-used.jinja').write_text(policy.tokenizer.chat_template)
    def save():
        (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        summary={}
        for arm in ('no_hint','manual_hint'):
            subset=[r for r in results if r['arm']==arm];summary[arm]={}
            for category in ('missing','copy','correction'):
                group=[r for r in subset if r['category']==category];summary[arm][category]={'passed':sum(r['passed'] for r in group),'total':len(group)}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'conversations.txt').write_text('\n\n'.join('CASE: '+r['case_id']+' | ARM: '+r['arm']+' | SEED: '+str(r['seed'])+' | PASS: '+str(r['passed'])+'\nEXPECTED: '+json.dumps(r['expected'],ensure_ascii=False)+'\nHINT: '+r['manual_hint']+'\n\n'+'\n\n'.join(('TOOL RESPONSE / CUSTOMER' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'][1:]) for r in results))
    for case in cases:
        for arm in ('no_hint','manual_hint'):
            messages=copy.deepcopy(case['prompt']);check_system(messages[0]['content'],case['tools'])
            if arm=='manual_hint':messages[-1]['content']+='\n\nGuidance for your next response: '+case['manual_hint']+'\nCarry out this instruction now; do not discuss the guidance.'
            text=policy.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            assert text.count('<tools>')==text.count('</tools>')==1,'Duplicated rendered catalog'
            ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            assert len(ids)+256<=8192
            (rendered/(case['id']+'-'+arm+'.txt')).write_text(text)
            for seed in (42,43):
                generated=policy.generate({'prompt_ids':ids},256,seed);details=score_decision(generated['text'],case)
                missing=case['reward_reference'].get('decision')=='clarify'
                passed=bool(generated['ended_with_eos'] and (details.get('clarification_correct') if missing else details.get('exact_call_set') and details.get('schema_validity')==1 and details.get('format_valid')))
                results.append({'case_id':case['id'],'category':case['category'],'arm':arm,'seed':seed,'passed':passed,'manual_hint':case['manual_hint'] if arm=='manual_hint' else '', 'expected':case['reward_reference'],'output':generated['text'],'score':details,'messages':messages+[{'role':'assistant','content':generated['text']}],'prompt_tokens':len(ids),'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()});save();print(case['id'],arm,seed,passed,repr(generated['text']),flush=True)
    (out/'run.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'cases_sha256':hashlib.sha256(source).hexdigest(),'temperature':.7,'top_p':1.,'seeds':[42,43],'completed':len(results),'optimizer_updates':0,'hint_placement':'Appended to latest user message as next-response instruction.','limitations':['Controlled diagnostic set, not held-out benchmark performance.','Clarification labels use a targeted-request heuristic; inspect actual wording.','No APIs executed; supplied tool histories are fixtures.','Manual hints use known task structure; this does not validate autonomous Qwen hints or distillation.']},indent=2))

if __name__=='__main__':main()
