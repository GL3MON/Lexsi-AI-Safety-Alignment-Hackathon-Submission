"""Causal Glaive clarification episodes with a deterministic customer actor."""
import copy
import json
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.glaive_signals import score_decision


def episode_rows(rows):
    followups={r['source_id']:r for r in rows if r['scenario']=='clarification_followup'}
    result=[]
    for original in rows:
        if original['scenario']=='clarification_followup':continue
        row=copy.deepcopy(original)
        if row['scenario']=='missing_required':
            follow=followups.get(row['source_id'])
            if follow is None or follow['group_id']!=row['group_id']:
                raise ValueError('Missing or mismatched customer followup '+row['id'])
            if follow['prompt'][:len(row['prompt'])]!=row['prompt']:
                raise ValueError('Followup opening differs from clarification prompt')
            row['customer_reply']=copy.deepcopy(follow['prompt'][-1])
            row['final_reference']=copy.deepcopy(follow['reward_reference'])
        result.append(row)
    return result



def tool_feedback(text, row, details, information_missing=False):
    """Report verification, never pretend a reference call executed a real API."""
    if '<tool_call' not in text:return None
    try:calls=parse_calls(text)
    except ValueError:calls=[]
    verified=(not information_missing and bool(details.get('exact_call_set'))
              and details.get('schema_validity')==1. and bool(details.get('format_valid')))
    payload={'name':calls[0]['name'] if len(calls)==1 else '__call_verifier__',
             'content':{'call_verified':verified,'api_executed':False,
                        'source':'deterministic_reference_verifier'}}
    if verified:
        payload['content']['status']='reference_call_matches'
        # No fabricated token, balance, receipt or other backend result.
    else:
        payload['content']['error']='missing_required_information' if information_missing else 'call_failed_verification'
    return {'role':'user','content':'<tool_response>'+json.dumps(payload)+'</tool_response>'}


def rollout(policy,cfg,row,seed,episode_id,with_reference=False,turn_hints=None):
    messages=copy.deepcopy(row['prompt']);actions=[];checks=[]
    clarification=row['scenario']=='missing_required'
    supplied=False;reason='turn_limit';complete=True;reward=0.
    for turn in range(min(cfg.max_actions,2 if clarification else 1)):
        policy_messages=copy.deepcopy(messages)
        hint=(turn_hints or {}).get(turn+1,'')
        if hint:
            policy_messages[0]['content']+='\n\nPRIVILEGED HINT FOR THIS ASSISTANT TURN ONLY:\n'+hint
        ids=policy.tokenizer.apply_chat_template(policy_messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        if len(ids)+cfg.response_tokens>cfg.context_limit:
            complete=False;reason='context_overflow';break
        generated=policy.generate({'prompt_ids':ids},cfg.response_tokens,seed+turn*100003)
        action={**generated,'prompt_ids':ids,'policy_version':policy.version,'turn_hint':hint}
        if with_reference:
            action['replay_old_logprobs']=policy.score(action).detach().cpu().tolist()
            action['reference_logprobs']=policy.score(action,reference=True).detach().cpu().tolist()
        actions.append(action);messages.append({'role':'assistant','content':generated['text']})
        if not generated['ended_with_eos']:
            reason='truncated_response';reward=0.;break
        reference={'tools':row['tools'],'reward_reference':row['final_reference']} if supplied else row
        details=score_decision(generated['text'],reference);checks.append(details)
        feedback=tool_feedback(generated['text'],row,details,information_missing=clarification and not supplied)
        if feedback:messages.append(feedback)
        if clarification and not supplied:
            if not details.get('clarification_correct'):
                reason='failed_to_clarify';reward=0.;break
            reward=.25
            if turn+1>=cfg.max_actions:
                reason='turn_limit';break
            # Only release the fixture value after the actual generated request
            # passes the targeted clarification check. Never inject gold assistant text.
            messages.append(copy.deepcopy(row['customer_reply']));supplied=True
        else:
            reward=(.25+.75*details['reward']) if clarification else details['reward']
            reason='correct_call' if details.get('exact_call_set') else 'incorrect_final_decision'
            break
    final=checks[-1] if checks else {}
    return {'task_id':row['id'],'episode_id':str(episode_id),'policy_version':policy.version,
            'complete':complete,'reward':reward if complete else None,
            'reward_details':{**final,'reward':reward,'clarification_correct':float(clarification and supplied),
                              'step_checks':checks,'reward_version':'glaive-episode-v1'},
            'actions':actions,'prompt':row['prompt'],'messages':messages,'scenario':row['scenario'],
            'termination_reason':reason,'customer_replied':supplied,'agent_turns':len(actions),
            'note':'Deterministic customer fixture; tool responses report reference verification only, not API execution.'}
