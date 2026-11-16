"""Matched before/after next-action and connected clarification diagnostics."""
import gc,json,re
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows,rollout
from scripts.compare_full_tool_controlled import build_cases,score


def main():
    torch.set_num_threads(8)
    out=Path('outputs/diagnostics/glaive-rl-checkpoint16-comparison');out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()]
    case=next(r for r in episode_rows(rows) if r['scenario']=='missing_required' and r['reward_reference']['missing_field']=='email')
    system=json.loads(Path('outputs/retail-eval/three-test-tasks-three-sft-stages/short-2k/task-38.json').read_text())['messages'][0]
    system['content']+='\n\n'+Path('templates/glaive_decision_rules.txt').read_text()
    results=[]
    def save():
        (out/'results.json').write_text(json.dumps(results,indent=2))
        summaries={}
        texts=[]
        for label in ('sft-475','rl-16'):
            subset=[r for r in results if r['model']==label];cats={}
            for cat in sorted({r['category'] for r in subset}):
                items=[r for r in subset if r['category']==cat];cats[cat]={'passed':sum(r['passed'] for r in items),'total':len(items)}
            summaries[label]=cats
            for r in subset:
                texts.append('MODEL: '+label+' | CASE: '+r['id']+' | SEED: '+str(r['seed'])+' | PASS: '+str(r['passed'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'] if m['role']!='system'))
        (out/'summary.json').write_text(json.dumps(summaries,indent=2));(out/'conversations.txt').write_text('\n\n'.join(texts))
    for label,adapter in [('sft-475','outputs/sft-glaive/run-d9133530a1/checkpoint-475'),('rl-16','outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy')]:
        cfg=RetailRLConfig(initial_adapter=adapter,temperature=.7,top_p=.95,max_actions=10,response_tokens=256)
        policy=LocalPolicy(cfg)
        for c in build_cases():
            messages=[system]+c['messages'];ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            result=policy.generate({'prompt_ids':ids},256,42);passed,details=score(c,result['text'])
            results.append({'model':label,'id':c['id'],'seed':42,'category':c['category'],'passed':bool(passed),'output':result['text'],'details':details,'messages':messages+[{'role':'assistant','content':result['text']}],'checkpoint':adapter});save();print(label,c['id'],passed,flush=True)
        for seed in range(42,46):
            e=rollout(policy,cfg,case,seed,str(seed))
            results.append({'model':label,'id':'connected_missing_email','seed':seed,'category':'connected_episode','passed':e['termination_reason']=='correct_call','reward':e['reward'],'clarified':e['customer_replied'],'messages':e['messages'],'checkpoint':adapter});save();print(label,'episode',seed,e['reward'],flush=True)
        del policy;gc.collect();torch.cuda.empty_cache()
    (out/'DONE').write_text('Matched diagnostics complete; no optimizer updates.\n')

if __name__=='__main__':main()
