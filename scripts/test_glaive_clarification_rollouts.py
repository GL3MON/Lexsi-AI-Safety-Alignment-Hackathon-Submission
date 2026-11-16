"""Independent first-response probes of the missing-email clarification fixture."""
import json
import argparse
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_signals import score_decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--explicit-clarification",action="store_true");args=parser.parse_args()
    torch.set_num_threads(8)
    rows=[json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()]
    row=next(r for r in rows if r['scenario']=='missing_required' and r['reward_reference']['missing_field']=='email')
    if args.explicit_clarification:
        row['prompt'][0]['content'] += '\n\n'+Path('templates/glaive_decision_rules.txt').read_text()
    cfg=RetailRLConfig(initial_adapter='outputs/sft-glaive/run-d9133530a1/checkpoint-475',temperature=.7,top_p=.95)
    policy=LocalPolicy(cfg)
    ids=policy.tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,return_dict=False)
    out=Path('outputs/diagnostics/glaive-clarification-explicit-prompt-12-rollouts' if args.explicit_clarification else 'outputs/diagnostics/glaive-clarification-12-rollouts');out.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in range(42,54):
        result=policy.generate({'prompt_ids':ids},256,seed)
        scores=score_decision(result['text'],row)
        results.append({'seed':seed,'output':result['text'],'scores':scores,'ended_with_eos':result['ended_with_eos']})
        report={'checkpoint':cfg.initial_adapter,'temperature':.7,'top_p':.95,'explicit_clarification_prompt':args.explicit_clarification,'case_id':row['id'],'prompt':row['prompt'],'rollouts':results,'clarifications':sum(r['scores']['clarification_correct'] for r in results),'completed':len(results),'target':12,'note':'Identical missing-email prompt; independent seeds; no reference answer in model prompt; no tools executed.'}
        (out/'results.json').write_text(json.dumps(report,indent=2))
        (out/'conversations.txt').write_text('CHECKPOINT: '+cfg.initial_adapter+'\nTEMPERATURE: 0.7 | TOP_P: 0.95\n\n'+'\n\n'.join(m['role'].upper()+': '+m['content'] for m in row['prompt'])+'\n\n'+'\n\n'.join('ROLLOUT seed='+str(r['seed'])+' | CLARIFIED='+str(bool(r['scores']['clarification_correct']))+'\nASSISTANT: '+r['output'] for r in results))
        print(json.dumps(results[-1]),flush=True)
    print('SUMMARY',report['clarifications'],'/',report['completed'],flush=True)

if __name__=='__main__':main()
