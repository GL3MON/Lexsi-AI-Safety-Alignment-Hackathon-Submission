"""Episode-balanced clipped GRPO on exact action contexts, not flattened segments."""
import math
import json


def group_advantages(episodes):
    if len(episodes)<2: raise ValueError('Need at least two independent episodes')
    if len({str(e['task_id']) for e in episodes})!=1 or len({e['policy_version'] for e in episodes})!=1:
        raise ValueError('Group must share one task and behavior policy version')
    if any(not e.get('complete') or e.get('reward') is None or not math.isfinite(e['reward']) for e in episodes):
        raise ValueError('Incomplete/failed episodes cannot be assigned GRPO advantages')
    if len({e['episode_id'] for e in episodes})!=len(episodes): raise ValueError('Duplicate episode in group')
    rewards=[e['reward'] for e in episodes];mean=sum(rewards)/len(rewards)
    std=math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+1e-8) for r in rewards]


def clipped_token_loss(new,old,reference,advantage,epsilon,beta,behavior=None):
    import torch
    ratio=torch.exp(new-old)
    surrogate=torch.minimum(ratio*advantage,torch.clamp(ratio,1-epsilon,1+epsilon)*advantage)
    delta=reference-new
    kl=torch.exp(delta)-delta-1
    # Separate PPO policy-change clipping from the numerical sampler/replay gap.
    # Without clipping: exp(new-old)*exp(old-behavior) == exp(new-behavior).
    correction=torch.exp(old-behavior).detach() if behavior is not None else 1.0
    return correction*(-surrogate+beta*kl)


def update_group(policy,optimizer,episodes,cfg):
    import torch
    if cfg.temperature<=0 or cfg.top_p!=1.0 or any(a.get('decoding_mode')=='greedy' or a.get('top_p',1.0)!=1.0 for e in episodes for a in e.get('actions', [])):
        raise ValueError('Greedy/nucleus evaluation episodes cannot be used for GRPO updates')
    advantages=group_advantages(episodes)
    if any(e['policy_version']!=policy.version for e in episodes): raise ValueError('Stale behavior policy group')
    # An all-tie group contains no task-reward learning signal. Do not manufacture it.
    if not any(advantages): return dict(skipped_tied_group=True,loss=0.0,optimizer_steps=0)
    largest_error=0.0;sampling_gap=0.0;last_loss=0.0;grad_norm=0.0
    for update in range(cfg.updates_per_group):
        optimizer.zero_grad(set_to_none=True);total=0.0;processed_actions=0
        print(json.dumps(dict(event='rl_gradient_start',actions=sum(len(e['actions']) for e in episodes),update=update)),flush=True)
        try:
            for episode,advantage in zip(episodes,advantages,strict=True):
                count=sum(len(a['completion_ids']) for a in episode['actions'])
                if not count: raise ValueError('Episode has no generated policy tokens')
                for action in episode['actions']:
                    if action['policy_version']!=episode['policy_version']: raise ValueError('Mixed action policy versions')
                    new=policy.score(action,grad=True)
                    behavior=torch.tensor(action['old_logprobs'],device=new.device,dtype=new.dtype)
                    old=torch.tensor(action.get('replay_old_logprobs',action['old_logprobs']),device=new.device,dtype=new.dtype)
                    ref=torch.tensor(action['reference_logprobs'],device=new.device,dtype=new.dtype)
                    if new.shape!=old.shape or new.shape!=ref.shape or new.shape!=behavior.shape: raise ValueError('Token probabilities are not aligned')
                    sampling_gap=max(sampling_gap,float((old-behavior).abs().max()))
                    if update==0:
                        error=float((new.detach()-old).abs().max());largest_error=max(largest_error,error)
                        if error>cfg.replay_tolerance: raise RuntimeError(f'Behavior/replay logprob mismatch {error:.6f} > {cfg.replay_tolerance}')
                    loss=clipped_token_loss(new,old,ref,advantage,cfg.clip_epsilon,cfg.kl_beta,behavior).sum()/(count*len(episodes))
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite policy loss')
                    total+=float(loss.detach());loss.backward();processed_actions+=1
                    if processed_actions%10==0:print(json.dumps(dict(event='rl_gradient_progress',completed_actions=processed_actions,update=update)),flush=True)
            parameters=[p for group in optimizer.param_groups for p in group['params']]
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg.max_grad_norm,error_if_nonfinite=True)
            grad_norm=float(norm);optimizer.step();last_loss=total
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
    policy.version+=1
    return dict(loss=last_loss,grad_norm=grad_norm,max_policy_replay_error=largest_error,
                max_sampling_replay_gap=sampling_gap,
                optimizer_steps=cfg.updates_per_group,advantages=advantages,skipped_tied_group=False)
