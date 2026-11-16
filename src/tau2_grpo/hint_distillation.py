"""Independent hindsight teacher contexts and top-k-plus-tail JSD."""
import copy
import json


def teacher_messages(history,observation,outcome,hint):
    """Append feedback to the last user turn; never mutate student history."""
    messages=copy.deepcopy(history)
    if not messages or messages[-1]['role']!='user':raise ValueError('Expected a pre-action user/observation turn')
    feedback=('\n\nFeedback on your earlier attempt:\nOutcome: '+json.dumps(outcome)+
              '\nTool observation: '+json.dumps(observation,ensure_ascii=False)+
              '\nReflection: '+(hint if isinstance(hint,str) else json.dumps(hint,ensure_ascii=False))+
              '\nUse this feedback when evaluating the original action for this turn.')
    messages[-1]['content']+=feedback
    return messages


def topk_tail_jsd(student_logits,teacher_logits,k=100):
    """Full-softmax top-k student support, one tail bucket; teacher detached."""
    import torch
    import torch.nn.functional as F
    student=F.log_softmax(student_logits.float(),dim=-1)
    teacher=F.log_softmax(teacher_logits.detach().float(),dim=-1)
    indices=student.detach().topk(min(k,student.shape[-1]),dim=-1).indices
    p=student.gather(-1,indices).exp();q=teacher.gather(-1,indices).exp()
    p=torch.cat((p,(1-p.sum(-1,keepdim=True)).clamp_min(1e-8)),dim=-1)
    q=torch.cat((q,(1-q.sum(-1,keepdim=True)).clamp_min(1e-8)),dim=-1)
    p=p/p.sum(-1,keepdim=True);q=q/q.sum(-1,keepdim=True)
    mixture=((p+q)/2).clamp_min(1e-30)
    divergence=.5*((p*(p.clamp_min(1e-30).log()-mixture.log())).sum(-1)+
                    (q*(q.clamp_min(1e-30).log()-mixture.log())).sum(-1))
    return divergence


def completion_hidden(model,prompt_ids,completion_ids):
    import torch
    ids=torch.tensor([prompt_ids+completion_ids],device=next(model.parameters()).device)
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    hidden=base.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
    return hidden[0,len(prompt_ids)-1:-1],base


def turn_distillation_loss(policy,student_ids,teacher_ids,completion_ids,grad=False,k=100,temperature=1.0):
    """Compare distributions at the exact original sampled completion prefixes."""
    import torch
    if temperature<=0: raise ValueError('Distillation temperature must be positive')
    # Frozen reference forward first; no graph/teacher gradients retained.
    with policy.reference(),torch.no_grad():
        policy.model.eval();teacher_hidden,_=completion_hidden(policy.model,teacher_ids,completion_ids)
        teacher_hidden=teacher_hidden.detach()
    policy.model.train(grad)
    with torch.set_grad_enabled(grad):
        student_hidden,base=completion_hidden(policy.model,student_ids,completion_ids)
        scale=getattr(base,'logit_scale',1.)
        # Recompute vocabulary projections during backward instead of retaining
        # a full-vocabulary softmax graph for every generated token.
        from torch.utils.checkpoint import checkpoint
        def score_pair(student_state,teacher_state):
            student_logits=(base.lm_head(student_state.unsqueeze(0))*scale).float()/temperature
            with torch.no_grad():
                teacher_logits=(base.lm_head(teacher_state.unsqueeze(0))*scale).float()/temperature
            return topk_tail_jsd(student_logits,teacher_logits,k).squeeze()
        parts=[]
        for index in range(len(completion_ids)):
            if grad:
                value=checkpoint(score_pair,student_hidden[index],teacher_hidden[index],use_reentrant=False)
            else:
                value=score_pair(student_hidden[index],teacher_hidden[index])
            parts.append(value)
        return torch.stack(parts)
