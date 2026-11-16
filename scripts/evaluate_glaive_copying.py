"""Checkpoint snapshot diagnostics; no changes to ongoing SFT."""
import json
import os
from pathlib import Path
import torch
from scripts.compare_full_tool_controlled import build_cases, score
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy

def main():
    torch.set_num_threads(8)
    out=Path(os.environ.get('GLAIVE_EVAL_OUTPUT','outputs/diagnostics/glaive-checkpoint400-copying'))
    cfg=RetailRLConfig(model_id='GL3MON/tiny-aya-global-function-calling-sft', initial_adapter=str(out/'adapter'), device=os.environ.get('GLAIVE_EVAL_DEVICE','cpu'), temperature=.7, top_p=.95)
    policy=LocalPolicy(cfg)
    system=json.loads(Path('outputs/retail-eval/three-test-tasks-three-sft-stages/short-2k/task-38.json').read_text())['messages'][0]
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    def save():
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        (out/'conversations.txt').write_text('\n\n'.join('CASE: '+r['id']+' | temperature='+str(r['temperature'])+' | PASS='+str(r['passed'])+'\nEXPECTED: '+json.dumps(r['expected_call'] or r['question'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages']) for r in rows))
        summary={}
        for temp in (0,.7):
            subset=[r for r in rows if r['temperature']==temp]
            summary[str(temp)]={c:{'passed':sum(r['passed'] for r in subset if r['category']==c),'total':sum(r['category']==c for r in subset)} for c in sorted({r['category'] for r in subset})}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'config.json').write_text(json.dumps({'checkpoint':os.environ.get('GLAIVE_EVAL_CHECKPOINT','outputs/sft-glaive/run-d9133530a1/checkpoint-400'),'base':cfg.model_id,'temperatures':[0,.7],'top_p':.95,'seed':42,'max_new_tokens':256,'device':cfg.device,'note':'Independent next-action diagnostics with full 17-tool retail prompt, not end-to-end task rewards.'},indent=2))
    for temp in (0,.7):
        policy.cfg=RetailRLConfig(**{**cfg.__dict__,'temperature':temp})
        for case in build_cases():
            if any(r['id']==case['id'] and r['temperature']==temp for r in rows): continue
            messages=[system]+case['messages']
            ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            result=policy.generate({'prompt_ids':ids},256,42)
            passed,details=score(case,result['text'])
            rows.append({**case,'temperature':temp,'passed':passed,'output':result['text'],'messages':case['messages']+[{'role':'assistant','content':result['text']}],'ended_with_eos':result['ended_with_eos'],**details})
            save(); print(temp,case['id'],passed,repr(result['text']),flush=True)
    (out/'DONE').write_text('24 probes complete\n')
if __name__=='__main__':main()
