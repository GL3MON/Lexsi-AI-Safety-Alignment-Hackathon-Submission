#!/usr/bin/env python3
"""Three-example continued-adapter overfit with identical greedy before/after probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/diagnostics/retail-clarification-overfit-v1'
INITIAL = 'outputs/sft-retail-corrective-v2/run-f245fba421/final_adapter'


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def probe(stage, adapter):
    import torch
    from tau2_grpo.rl_config import RetailRLConfig
    from tau2_grpo.rl_policy import LocalPolicy
    cfg = RetailRLConfig(initial_adapter=adapter, temperature=0.0)
    policy = LocalPolicy(cfg)
    results = []
    for case in json.loads((OUT/'cases.json').read_text()):
        messages = case['messages']
        prompt = policy.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
        full = policy.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        assert full[:len(prompt)] == prompt
        gold = full[len(prompt):]
        score = policy.score({'prompt_ids':prompt, 'completion_ids':gold})
        nll = -score.mean().item()
        del score
        ids = torch.tensor([prompt], device=cfg.device)
        with torch.inference_mode():
            logits = policy.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[0,-1].float()
            probs = logits.softmax(-1)
            gold_prob = probs[gold[0]].item()
            values, indices = probs.topk(5)
            top = [{'token':policy.tokenizer.decode([i]),'probability':v} for i,v in zip(indices.tolist(),values.tolist())]
        del ids, logits, probs
        output = policy.generate({'prompt_ids':prompt},128,42)['text']
        expected = messages[-1]['content']
        asks = 'email' in output.lower() and any(s in output.lower() for s in ('provide','share','could you','what is','may i','can you')) and '<tool_call>' not in output
        result = dict(task_id=case['task_id'], split=case['diagnostic_split'], expected=expected, output=output, exact_match=output.strip()==expected.strip(), asks_for_email=asks, gold_nll=nll, first_gold_token_probability=gold_prob, first_top5=top, prompt_tokens=len(prompt), target_tokens=len(gold))
        results.append(result)
        print(json.dumps(result), flush=True)
    write(OUT/f'{stage}.json',results)
    (OUT/f'{stage}.txt').write_text('\n\n'.join(f"Task {r['task_id']} ({r['split']})\nEXPECTED: {r['expected']}\nGENERATED: {r['output']}" for r in results))


def main():
    from transformers import AutoTokenizer
    from tau2_grpo.sft_data import tokenize_assistant_only
    OUT.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'outputs/hf-export/tiny-aya-global-function-calling-sft-525')
    cases=[]
    used=set()
    for line in (ROOT/'outputs/retail-data/retail-sft-corrective-v2-final/sft.jsonl').read_text().splitlines():
        r=json.loads(line)
        if str(r['task_id']) in used or r.get('source_terminal_reward') != 1: continue
        index=next((i for i,m in enumerate(r['messages']) if m['role']=='assistant'),None)
        if index is None or index not in r['loss_message_indices']: continue
        text=r['messages'][index]['content'].lower()
        if 'email' not in text or '<tool_call>' in text or not any(x in text for x in ('provide','share','could you','what is')): continue
        r['messages']=r['messages'][:index+1]
        r['loss_message_indices']=[index]
        r['id']='overfit-clarification-'+str(r['task_id'])
        r['diagnostic_split']='fit' if len(cases)<3 else 'control'
        encoded,spans=tokenize_assistant_only(tokenizer,r['messages'],8192,[index])
        assert len(spans)==1 and any(x!=-100 for x in encoded['labels'])
        print('Selected',r['task_id'],len(encoded['input_ids']),r['messages'][-1]['content'],flush=True)
        cases.append(r);used.add(str(r['task_id']))
        if len(cases)==5:break
    assert len(cases)==5
    write(OUT/'cases.json',cases)
    for name in ('fit','control'):
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases if r['diagnostic_split']==name))
    cfg=json.loads((ROOT/'configs/sft-retail.json').read_text())
    cfg.update(initial_adapter=INITIAL,dataset_path=str(OUT/'fit.jsonl'),eval_dataset_path=str(OUT/'fit.jsonl'),task_split_manifest=None,rl_holdout_path=None,output_dir=str(OUT/'training'),overfit_debug=True,num_train_epochs=20,max_steps=30,batch_size=1,gradient_accumulation_steps=1,learning_rate=2e-4,lr_scheduler_type='constant',warmup_ratio=0.,weight_decay=0.,eval_steps=10,save_steps=10,load_best_model_at_end=False,evaluate_before_training=True,log_sample_count=3,sample_max_new_tokens=128,wandb_run_name='clarification-overfit-3-examples',wandb_tags=['overfit-diagnostic','continued-retail-adapter'])
    write(OUT/'sft-config.json',cfg)
    env=dict(os.environ,HF_HUB_OFFLINE='1')
    def run(args,log):
        print('Running',args,flush=True)
        with (OUT/log).open('w') as f:
            subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    run([__file__,'--probe','before','--adapter',INITIAL],'before.log')
    run(['-m','tau2_grpo.sft','--config',str(OUT/'sft-config.json')],'training.log')
    adapters=list((OUT/'training').glob('run-*/final_adapter'))
    assert len(adapters)==1,adapters
    run([__file__,'--probe','after','--adapter',str(adapters[0])],'after.log')
    before=json.loads((OUT/'before.json').read_text());after=json.loads((OUT/'after.json').read_text())
    report={'adapter':str(adapters[0]),'note':'Controls excluded from this diagnostic fit, but drawn from the original training corpus; not benchmark holdout.','before':before,'after':after}
    for stage,rows in [('before',before),('after',after)]:
        report[stage+'_counts']={key:sum(r[key] for r in rows if r['split']=='fit') for key in ('exact_match','asks_for_email')}
    write(OUT/'report.json',report)
    import wandb
    run=wandb.init(project='tau2-retail',job_type='overfit-comparison',name='clarification-overfit-before-after',mode='online',config={'fit_examples':3,'control_examples':2,'controls_are_benchmark_holdout':False,'adapter':str(adapters[0])})
    run.log({'generations':wandb.Table(columns=['stage','split','task','expected','output','nll','first_token_probability'],data=[[stage,r['split'],str(r['task_id']),r['expected'],r['output'],r['gold_nll'],r['first_gold_token_probability']] for stage,rows in [('before',before),('after',after)] for r in rows])})
    report['comparison_wandb_url']=run.url
    write(OUT/'report.json',report)
    run.finish()
    print('DONE',json.dumps({k:v for k,v in report.items() if k not in ('before','after')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',choices=['before','after']);p.add_argument('--adapter');a=p.parse_args()
    if a.probe: probe(a.probe,a.adapter)
    else: main()
