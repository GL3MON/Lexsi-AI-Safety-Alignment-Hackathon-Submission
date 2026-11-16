"""One continuous Tau2 episode across any number of summary boundaries."""
import copy
import json
import math
import traceback
from pathlib import Path
from uuid import uuid4


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def collect_episode(policy,env,controller,task_id,seed,cfg,output,with_reference=False):
    record=dict(episode_id=uuid4().hex,task_id=str(task_id),seed=seed,policy_version=policy.version,
                reward=None,complete=False,messages=[],actions=[],summaries=[],external=[])
    output=Path(output)
    from tau2_grpo.rl_rewards import ToolRewardTracker,shaped_reward
    tracker=ToolRewardTracker(cfg) if cfg.reward_mode=='additive_grounded' else None
    def save():
        record['summaries']=copy.deepcopy(controller.summaries)
        record['summary_attempts']=copy.deepcopy(getattr(getattr(controller,'summarizer',None),'attempts',[]))
        if hasattr(env,'external'): record['external']=copy.deepcopy(env.external.audit)
        if tracker is not None:record['tool_reward_audit']=copy.deepcopy(tracker.events)
        atomic_json(output,record)
    try:
        raw=env.reset();record['messages']=raw
        if hasattr(env,'assistance'):
            record['assistance']=copy.deepcopy(env.assistance)
            if tracker is not None:
                from tau2_grpo.assisted_tau2 import prime_tracker
                tracker.prefix_schemas=env.schemas;prime_tracker(tracker,raw)
        save();generated=0
        for turn in range(cfg.max_actions):
            if generated>=cfg.max_episode_tokens:
                record['stop_reason']='episode_token_budget';break
            prepared=controller.prepare(raw,seed+turn)
            result=policy.generate(prepared,min(cfg.response_tokens,cfg.max_episode_tokens-generated),seed+turn)
            tokens=result['completion_ids'];logs=result['old_logprobs']
            if not tokens or len(tokens)!=len(logs) or any(not math.isfinite(v) for v in logs):
                raise RuntimeError('Missing or invalid sampled-token probabilities')
            if len(prepared['prompt_ids'])+len(tokens)>cfg.context_limit:
                raise RuntimeError('Actor exceeded its supported context')
            action=dict(**prepared,**result,turn=turn,policy_version=policy.version,
                        raw_message_count=len(raw),summary_count=len(controller.summaries),temperature=cfg.temperature)
            if hasattr(policy, 'score'):
                action['replay_old_logprobs']=policy.score(action).detach().cpu().tolist()
                if len(action['replay_old_logprobs']) != len(tokens) or any(not math.isfinite(v) for v in action['replay_old_logprobs']):
                    raise RuntimeError('Replay probabilities are not token-aligned')
                if action.get('decoding_mode') != 'greedy':
                    action['max_sampling_replay_gap']=max(abs(a-b) for a,b in zip(action['old_logprobs'],action['replay_old_logprobs'],strict=True))
            if with_reference:
                action['reference_logprobs']=policy.score(action,reference=True).detach().cpu().tolist()
                if len(action['reference_logprobs']) != len(tokens) or any(not math.isfinite(v) for v in action['reference_logprobs']):
                    raise RuntimeError('Reference probabilities are invalid')
            record['actions'].append(action);generated+=len(tokens)
            raw.append({'role':'assistant','content':result['text']})
            save()
            print(json.dumps(dict(event='actor_action',episode_id=record['episode_id'],task_id=task_id,
                turn=turn,prompt_tokens=len(prepared['prompt_ids']),generated_tokens=len(tokens),
                summaries=len(controller.summaries),text=result['text']),ensure_ascii=False),flush=True)
            if not result.get('ended_with_eos',True):
                incoming=[dict(role='tool',name='__runtime__',content=json.dumps({'name':'__runtime__','content':{'error':'Response token limit reached. Produce a complete call or concise customer response.'}}))]
                done=False;call=None
                if cfg.environment in ('short_retail','short_bank'):
                    # An incomplete response is a failed short decision, not a
                    # prefix that can receive positive credit after a later retry.
                    incoming,done,call=env.step('')
            else:
                incoming,done,call=env.step(result['text'])
            if call:
                call_id=next((m.get('tool_call_id') for m in incoming if m['role']=='tool'),None) or 'action-'+str(turn)
                raw[-1]['tool_calls']=[{'id':call_id,'type':'function',
                    'function':{'name':call['name'],'arguments':copy.deepcopy(call['arguments'])}}]
            if tracker is not None:tracker.record(result['text'],raw[:-1],incoming,getattr(env,'schemas',{}))
            raw.extend(incoming);save()
            if done:
                record['stop_reason']='terminal';break
        else:
            record['stop_reason']='action_budget'
        reward,simulation=env.finish()
        if not math.isfinite(reward): raise RuntimeError('Nonfinite terminal reward')
        record['native_reward']=None if cfg.environment in ('short_retail','short_bank') else reward
        if cfg.environment in ('short_retail','short_bank'):
            record['short_task_success']=bool(simulation.get('short_success'))
            record['reward_details']=env.reward_details
        if tracker is not None:reward,record['reward_details']=shaped_reward(reward,simulation,tracker)
        record.update(reward=reward,complete=True,simulation=simulation,generated_tokens=generated,
                      forced_stop=record.get('stop_reason')!='terminal')
    except Exception as error:
        record.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        # Infrastructure/context preparation failures remain null, never zero labels.
        record['reward']=None;record['complete']=False
    finally:
        try: env.close()
        except Exception as error:
            record['cleanup_error']=str(error);record['complete']=False;record['reward']=None
        from tau2_grpo.rl_reporting import episode_check_report
        record['check_report']=episode_check_report(record)
        save()
    output.with_suffix('.txt').write_text('\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in record['messages']))
    return record
