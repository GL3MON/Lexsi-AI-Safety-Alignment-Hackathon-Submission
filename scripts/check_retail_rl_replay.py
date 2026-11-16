"""Audit saved exact contexts, sampler corrections, and optional GPU backprop."""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from tau2_grpo.rl_config import RetailRLConfig,path
from tau2_grpo.rl_policy import LocalPolicy,disable_dropout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--all-actions',action='store_true')
    parser.add_argument('--grad-check',action='store_true')
    args=parser.parse_args()
    saved=json.loads((args.run/'config.json').read_text())
    cfg=RetailRLConfig(**{k:v for k,v in saved.items() if k in asdict(RetailRLConfig())})
    policy=LocalPolicy(cfg,adapter=False)
    report={}
    for label in ['function_calling_base','retail_sft']:
        if label=='retail_sft':
            from peft import PeftModel
            policy.model=PeftModel.from_pretrained(policy.model,str(path(cfg.initial_adapter)),adapter_name='policy',is_trainable=args.grad_check)
            disable_dropout(policy.model);policy.model.eval()
        episode=json.loads((args.run/(label+'.json')).read_text())
        actions=episode['actions']
        chosen=actions if args.all_actions else [actions[0]]+[a for a in actions if a['summary_count']][:1]
        checks=[]
        for a in chosen:
            assert policy.tokenizer.encode(a['rendered_prompt'],add_special_tokens=False)==a['prompt_ids']
            assert policy.tokenizer.decode(a['completion_ids'],skip_special_tokens=True)==a['text']
            new=policy.score(a).detach().cpu().tolist()
            old=a['replay_old_logprobs']
            error=max(abs(x-y) for x,y in zip(new,old,strict=True))
            gap=max(abs(x-y) for x,y in zip(old,a['old_logprobs'],strict=True))
            checks.append(dict(turn=a['turn'],summary_count=a['summary_count'],tokens=len(new),
                max_policy_replay_error=error,max_sampling_replay_gap=gap,
                passed=error<=cfg.replay_tolerance and all(math.isfinite(v) for v in new)))
        report[label]=checks
        print(json.dumps({label:checks}),flush=True)
    result=dict(checks=report,tolerance=cfg.replay_tolerance,passed=all(c['passed'] for v in report.values() for c in v))
    if args.grad_check:
        import torch
        policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        policy.model.enable_input_require_grads()
        action=max(actions,key=lambda a:len(a['prompt_ids'])+len(a['completion_ids']))
        if cfg.device=='cuda':torch.cuda.reset_peak_memory_stats()
        replay=policy.score(action,grad=True)
        expected=torch.tensor(action['replay_old_logprobs'],device=replay.device)
        error=float((replay.detach()-expected).abs().max())
        (-replay.mean()).backward()
        gradients=[p.grad for p in policy.model.parameters() if p.requires_grad and p.grad is not None]
        finite=bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
        result['gradient_check']=dict(turn=action['turn'],prompt_tokens=len(action['prompt_ids']),
            completion_tokens=len(action['completion_ids']),max_policy_replay_error=error,
            finite_gradients=finite,nonzero_gradients=any(bool(g.abs().sum()>0) for g in gradients),
            optimizer_steps=0,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cfg.device=='cuda' else None)
        result['passed'] &= finite and result['gradient_check']['nonzero_gradients'] and error<=cfg.replay_tolerance
        print(json.dumps(result['gradient_check']),flush=True)
    (args.run/'replay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit('Policy replay mismatch or invalid gradients; optimizer must remain blocked')


if __name__=='__main__':main()
