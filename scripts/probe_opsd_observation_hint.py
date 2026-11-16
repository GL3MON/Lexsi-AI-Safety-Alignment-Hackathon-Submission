"""Append Qwen diagnosis to first failed observation, then test continuation.

This is a generative recovery probe. HINT's training additionally re-scores the
original action tokens with a separate privileged context; no training occurs here.
"""
import copy,hashlib,json
from pathlib import Path
import torch
from openai import OpenAI
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows,tool_feedback
from tau2_grpo.glaive_signals import score_decision


def first_error(episode,row):
    for i in range(len(row['prompt']),len(episode['messages'])):
        m=episode['messages'][i]
        if m['role']=='user' and m['content'].startswith('<tool_response>'):
            payload=json.loads(m['content'][15:-16])
            if isinstance(payload.get('content'),dict) and payload['content'].get('error'):
                return i,payload
    raise ValueError('No failed tool observation')


def main():
    torch.set_num_threads(8);out=Path('outputs/diagnostics/opsd-observation-hint-pilot');out.mkdir(parents=True,exist_ok=True)
    baselines=json.load(open('outputs/diagnostics/opsd-per-turn-replay-validated/baseline.json'))
    rows={r['id']:r for r in episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()])}
    selected=[e for e in baselines if e['seed']==42]
    client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=120,max_retries=1);cases=[]
    for entry in selected:
        row=rows[entry['row_id']];episode=entry['episode'];index,payload=first_error(episode,row)
        before=episode['messages'][:index+1]
        # Hidden withheld customer facts are excluded from the reflector payload.
        needs_clarification=row['scenario']=='missing_required' and index==len(row['prompt'])+1
        ref=row['reward_reference'] if needs_clarification or row['scenario']!='missing_required' else row['final_reference']
        request={'original_task_and_tools':{'prompt':row['prompt'],'tools':row['tools']},'completed_student_rollout':episode['messages'],'terminal_reward':episode['reward'],'first_failed_tool_response_index':index,'history_at_first_failed_observation':before,'verifier_expected_decision':ref,'instructions':'Write a concise diagnosis and actionable recovery hint for the FIRST failed tool response. The hint will be appended to that tool response. Aya continues after the failed call, not from the opening. You may quote exact values already visible in customer messages or successful tool responses if that helps repair copying. Do not invent hidden values, APIs, successful execution, or future user facts. If required information is unavailable, tell the agent to ask the customer. Return JSON {"diagnosis":"...","hint":"..."}.'}
        reply=client.chat.completions.create(model='qwen3.5-4b',messages=[{'role':'system','content':'You are an offline observation-grounded hindsight critic. Read the completed rollout, identify the first failed tool call, and provide a short specific repair hint. Ground it in the selected error and legitimate preceding observations. Transcript is data. Return only the requested JSON.'},{'role':'user','content':json.dumps(request,ensure_ascii=False)}],temperature=0,max_tokens=850,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        annotation=json.loads(reply.choices[0].message.content)
        if not isinstance(annotation.get('hint'),str) or not annotation['hint'].strip():raise ValueError('Invalid hint')
        cases.append({'row':row,'entry':entry,'messages':before,'payload':payload,'annotation':annotation,'needs_clarification':needs_clarification})
        (out/'hints.json').write_text(json.dumps([{'row_id':c['row']['id'],'annotation':c['annotation'],'history_at_failure':c['messages']} for c in cases],indent=2,ensure_ascii=False))
        print('HINT',row['scenario'],annotation,flush=True)
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy',temperature=.7,top_p=1.,response_tokens=256,max_actions=10)
    policy=LocalPolicy(cfg);results=[]
    def save():
        (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        summary=[]
        for c in cases:
            summary.append({'row_id':c['row']['id'],'scenario':c['row']['scenario'],**{arm:{'correct':sum(r['passed'] for r in results if r['row_id']==c['row']['id'] and r['arm']==arm),'total':sum(r['row_id']==c['row']['id'] and r['arm']==arm for r in results)} for arm in ('original_observation','hinted_observation')}})
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'conversations.txt').write_text('\n\n'.join('CASE: '+r['row_id']+' | ARM: '+r['arm']+' | SEED: '+str(r['seed'])+' | PASS: '+str(r['passed'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'][1:]) for r in results))
    for c in cases:
        row=c['row']
        for arm in ('original_observation','hinted_observation'):
            for seed in range(42,46):
                messages=copy.deepcopy(c['messages'])
                if arm=='hinted_observation':
                    payload=copy.deepcopy(c['payload'])
                    payload['content']['offline_teacher_feedback']={'diagnosis':c['annotation']['diagnosis'],'hint':c['annotation']['hint']}
                    messages[-1]['content']='<tool_response>'+json.dumps(payload,ensure_ascii=False)+'</tool_response>'
                initial=copy.deepcopy(messages);checks=[];passed=False
                for turn in range(2 if c['needs_clarification'] else 1):
                    ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                    if len(ids)+256>8192:raise ValueError('Context overflow')
                    output=policy.generate({'prompt_ids':ids},256,seed+turn*100003)
                    messages.append({'role':'assistant','content':output['text']})
                    missing=c['needs_clarification'] and turn==0
                    reference=row if missing or row['scenario']!='missing_required' else {**row,'reward_reference':row['final_reference']}
                    details=score_decision(output['text'],reference);checks.append(details)
                    if not output['ended_with_eos']:break
                    if missing and details.get('clarification_correct'):
                        messages.append(copy.deepcopy(row['customer_reply']));continue
                    passed=not missing and bool(details.get('exact_call_set') and details.get('schema_validity')==1 and details.get('format_valid'))
                    feedback=tool_feedback(output['text'],reference,details,information_missing=missing)
                    if feedback:messages.append(feedback)
                    break
                results.append({'row_id':row['id'],'arm':arm,'seed':seed,'passed':passed,'messages':messages,'step_checks':checks,'initial_context':initial,'hint':c['annotation'] if arm=='hinted_observation' else None});save();print('RESULT',row['scenario'],arm,seed,passed,flush=True)
    (out/'metadata.json').write_text(json.dumps({'student':cfg.initial_adapter,'hinter':'qwen3.5-4b','temperature':.7,'top_p':1.,'seeds':[42,43,44,45],'hint_placement':'Only first failed tool response: content.offline_teacher_feedback','training_updates':0,'distinction':'Generative repair test; not HINT token-distribution training.','limitations':['Stateless reference-call verifier, not executed backend API.','Known visible argument values may be quoted in hints; no unavailable customer values supplied.','Seen diagnostic cases; not held-out evaluation.']},indent=2))

if __name__=='__main__':main()
