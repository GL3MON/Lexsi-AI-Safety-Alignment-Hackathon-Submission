"""Compare exact seen SFT prompts on base/retail weights, without environment actions."""
import argparse
import json
import math
import os
from dataclasses import replace, asdict
from pathlib import Path

from tau2_grpo.rl_config import ROOT, load_settings, path
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.sft_data import read_records


def choose_cases(records):
    chosen={}
    for row in records:
        source=json.loads(Path(row['source_episode']).read_text())
        if source['record'].get('reward') != 1.0:
            continue
        selected=set(row['loss_message_indices'])
        first=next(i for i,m in enumerate(row['messages']) if m['role']=='assistant')
        for i in sorted(selected):
            text=row['messages'][i]['content']
            calls=parse_calls(text)
            kind=None
            if i==first and not calls and 'email' in text.lower() and any(w in text.lower() for w in ['provide','share','what is']):
                kind='seen_initial_clarification'
            elif len(calls)==1 and calls[0]['name']=='find_user_id_by_email' and '@' in row['messages'][i-1]['content']:
                kind='seen_email_lookup'
            elif len(calls)==1 and calls[0]['name']=='cancel_pending_order':
                kind='seen_cancellation'
            if kind and kind not in chosen:
                chosen[kind]=dict(kind=kind,record_id=row['id'],task_id=row['task_id'],
                    source_episode=row['source_episode'],message_index=i,
                    messages=row['messages'][:i],gold=text)
        if len(chosen)==3:break
    if len(chosen)!=3:raise ValueError('Could not locate all three diagnostic cases')
    return list(chosen.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.output.resolve();run.mkdir(parents=True,exist_ok=False)
    cfg=load_settings(ROOT/'configs/rl-retail.json',dict(temperature=0.,top_p=1.,group_size=1,response_tokens=128))
    rows=read_records(ROOT/'outputs/retail-data/retail-sft-final/sft.jsonl')
    cases=choose_cases(rows)
    (run/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    import torch
    import wandb
    from peft import PeftModel
    from tau2_grpo.rl_policy import LocalPolicy,disable_dropout
    wb=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=run.name,job_type='sft-diagnostic',dir=str(run),config=asdict(cfg))
    policy=LocalPolicy(cfg,adapter=False)
    results=[]
    success=False
    try:
        for label in ['function_calling_base','retail_sft']:
            if label=='retail_sft':
                policy.model=PeftModel.from_pretrained(policy.model,str(path(cfg.initial_adapter)),adapter_name='policy',is_trainable=False)
                disable_dropout(policy.model);policy.model.eval()
                (run/'adapter-state.json').write_text(json.dumps(dict(active_adapters=policy.model.active_adapters,
                    adapter=str(path(cfg.initial_adapter)),
                    lora_tensor_count=sum('lora_' in n for n,p in policy.model.named_parameters()),
                    nonzero_lora_b_tensors=sum(bool(p.detach().abs().sum()>0) for n,p in policy.model.named_parameters() if 'lora_B' in n)),indent=2)+'\n')
            for case in cases:
                prompt=policy.tokenizer.apply_chat_template(case['messages'],tokenize=False,add_generation_prompt=True)
                ids=policy.tokenizer.encode(prompt,add_special_tokens=False)
                complete=policy.tokenizer.apply_chat_template(case['messages']+[{'role':'assistant','content':case['gold']}],tokenize=True,return_dict=False,add_generation_prompt=False)
                if complete[:len(ids)]!=ids:raise ValueError('SFT prompt is not the exact gold prefix')
                gold_ids=complete[len(ids):]
                if len(ids)+max(128,len(gold_ids))>cfg.context_limit:raise ValueError('Diagnostic would exceed context')
                gold_score=policy.score(dict(prompt_ids=ids,completion_ids=gold_ids)).detach()
                generated=policy.generate(dict(prompt_ids=ids),128,cfg.seed)
                try: pred_calls=parse_calls(generated['text']);format_error=None
                except ValueError as error: pred_calls=[];format_error=str(error)
                gold_calls=parse_calls(case['gold'])
                answer=dict(model=label,kind=case['kind'],task_id=case['task_id'],record_id=case['record_id'],
                    gold=case['gold'],generated=generated['text'],
                    exact_text_match=generated['text'].strip()==case['gold'].strip(),
                    gold_calls=gold_calls,generated_calls=pred_calls,format_error=format_error,
                    exact_tool_calls_match=gold_calls==pred_calls if gold_calls else None,
                    gold_nll=float(-gold_score.mean()),prompt_tokens=len(ids),gold_tokens=len(gold_ids))
                results.append(answer)
                print(json.dumps(answer),flush=True)
                (run/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                lines=[]
                for r in results:
                    lines += [f"MODEL: {r['model']} / {r['kind']} / task {r['task_id']}",
                              f"Gold teacher-forced NLL: {r['gold_nll']:.6f}",
                              'EXPECTED:\n'+r['gold'],'GENERATED:\n'+r['generated'],'']
                (run/'results.txt').write_text('\n\n'.join(lines))
                wb.log({label+'/'+case['kind']+'/gold_nll':answer['gold_nll']})
        (run/'runtime.json').write_text(json.dumps(dict(wandb_url=wb.url,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            note='Three exact seen training prompts from reward-1 sources. Greedy generation. No task environment or optimizer updates. Text match is descriptive; paraphrases can be correct.'),indent=2)+'\n')
        success=True
    finally:wb.finish(exit_code=0 if success else 1)


if __name__=='__main__':main()
