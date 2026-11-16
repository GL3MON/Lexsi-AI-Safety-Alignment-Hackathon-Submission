"""HINT-style teacher-context scoring on exact student tokens; no updates."""
import json
from pathlib import Path
import torch
from openai import OpenAI
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy,action_logprobs
from tau2_grpo.hint_distillation import teacher_messages,turn_distillation_loss


def main():
    torch.set_num_threads(8);out=Path('outputs/diagnostics/hint-teacher-token-probe-v2');out.mkdir(parents=True,exist_ok=True)
    episodes=[e for e in json.load(open('outputs/diagnostics/opsd-per-turn-replay-validated/baseline.json')) if e['seed']==42]
    client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=120,max_retries=1)
    prepared=[]
    for item in episodes:
        e=item['episode'];cursor=len(e['prompt']);turns=[]
        for n,action in enumerate(e['actions'],1):
            position=next(i for i in range(cursor,len(e['messages'])) if e['messages'][i]['role']=='assistant' and e['messages'][i]['content']==action['text']);cursor=position+1
            observation=e['messages'][position+1] if position+1<len(e['messages']) else {'role':'user','content':'No subsequent environment observation recorded.'}
            turns.append({'turn':n,'history_before_turn':e['messages'][:position],'action':action,'observation':observation})
        reply=client.chat.completions.create(model='qwen3.5-4b',messages=[{'role':'system','content':'You are an observation-grounded hindsight critic for a multi-turn tool-use agent. Given a completed trajectory, its final reward and numbered generated turns, return one diagnosis per numbered turn. Return JSON {"turn1":{"diagnosis":"...","confidence":"high|medium|low","next_action_target":null}}. Diagnose correct turns too. A mistyped copy of an available value is a copying error, not missing information: prefer reusing the existing authoritative value instead of asking the customer to repeat it. A tool verification mismatch does not mean the original provided value is invalid. Ground each diagnosis in that action and immediate observation; later evidence may identify delayed consequences but do not quote raw future actions/observations. Distinguish agent decision errors from API execution errors. Set next_action_target only when an unambiguous correction is available from legitimate facts; otherwise null. Do not invent hidden customer facts. Only the explicitly numbered generated turns count, not prompt demonstrations.'},{'role':'user','content':json.dumps({'task':e['prompt'],'completed_trajectory':e['messages'],'outcome':e['reward'],'generated_turns':[{'turn':t['turn'],'action':t['action']['text'],'immediate_observation':t['observation']} for t in turns]},ensure_ascii=False)}],temperature=0,max_tokens=1300,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        annotations=json.loads(reply.choices[0].message.content)
        for t in turns:
            hint=annotations['turn'+str(t['turn'])]
            t.update(hint=hint,teacher_history=teacher_messages(t['history_before_turn'],t['observation'],e['reward'],hint),row_id=item['row_id'],episode_reward=e['reward'])
            prepared.append(t)
        (out/'prepared_turns.json').write_text(json.dumps(prepared,indent=2,ensure_ascii=False))
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy',temperature=1.)
    policy=LocalPolicy(cfg,trainable=True);results=[]
    for turn in prepared:
        student=policy.tokenizer.apply_chat_template(turn['history_before_turn'],tokenize=True,add_generation_prompt=True,return_dict=False)
        teacher=policy.tokenizer.apply_chat_template(turn['teacher_history'],tokenize=True,add_generation_prompt=True,return_dict=False)
        original=turn['action']['completion_ids']
        assert student==turn['action']['prompt_ids'],'Student context differs from actual rollout'
        assert len(teacher)+len(original)<=8192
        policy.model.eval()
        with torch.no_grad():
            student_logs=action_logprobs(policy.model,student,original,1.,16).cpu()
            with policy.reference():teacher_logs=action_logprobs(policy.model,teacher,original,1.,16).cpu()
            divergence=turn_distillation_loss(policy,student,teacher,original,grad=False,k=100).cpu()
        if not torch.isfinite(divergence).all():raise RuntimeError('Nonfinite JSD')
        tokens=[policy.tokenizer.decode([token],skip_special_tokens=False) for token in original]
        report={'row_id':turn['row_id'],'turn':turn['turn'],'hint':turn['hint'],'mean_jsd':float(divergence.mean()),'student_mean_original_logprob':float(student_logs.mean()),'teacher_mean_original_logprob':float(teacher_logs.mean()),'mean_teacher_minus_student':float((teacher_logs-student_logs).mean()),'tokens':[{'index':i,'id':token,'text':text,'student_logprob':float(s),'teacher_logprob':float(t),'teacher_minus_student':float(t-s),'jsd':float(j)} for i,(token,text,s,t,j) in enumerate(zip(original,tokens,student_logs,teacher_logs,divergence,strict=True))]}
        results.append(report)
        (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        (out/'teacher-contexts.txt').write_text('\n\n'.join('CASE: '+t['row_id']+' | TURN: '+str(t['turn'])+'\n\n'+'\n\n'.join(m['role'].upper()+': '+m['content'] for m in t['teacher_history'])+'\n\nORIGINAL ASSISTANT TOKENS TO SCORE:\n'+t['action']['text'] for t in prepared))
        (out/'token-scores.txt').write_text('\n\n'.join('CASE: '+r['row_id']+' | TURN: '+str(r['turn'])+' | MEAN JSD: '+str(r['mean_jsd'])+'\n'+'\n'.join(repr(t['text'])+' | student='+str(round(t['student_logprob'],4))+' | teacher='+str(round(t['teacher_logprob'],4))+' | delta='+str(round(t['teacher_minus_student'],4)) for t in r['tokens']) for r in results))
        print(json.dumps({k:v for k,v in report.items() if k!='tokens'},ensure_ascii=False),flush=True)
    (out/'metadata.json').write_text(json.dumps({'student':cfg.initial_adapter,'teacher':'Frozen identical checkpoint with privileged context','reflector':'MIG2 qwen3.5-4b','scored_tokens':'Exact saved student completions including EOS','distribution_temperature':1.,'support':'Student top100 plus tail bucket','optimizer_updates':0,'purpose':'Verify context construction and dense distillation signal before training.','limitations':['Stateless reference-verifier sandbox.','Saved student rollouts, no refresh or EMA updates in this probe.','No importance weighting applied because no policy update or train loss is performed.','No claim of behavioral improvement.']},indent=2))

if __name__=='__main__':main()
