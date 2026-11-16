"""Replay-checked, post-tool-error privileged self-distillation for Tau2 retail.

The first implementation deliberately accepts only explicit, nonterminal tool
errors with a subsequent agent decision. It does not guess recoverability from
terminal reward failures or silently incorrect database state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tau2_grpo.rl_config import ROOT, load_settings, path, task_ids
from tau2_grpo.rl_rollout import atomic_json


def read_config(filename):
    raw=json.loads(Path(filename).read_text())
    base_file=raw.pop('base_config',str(ROOT/'configs/rl-retail.json'))
    base=load_settings(base_file)
    extra={key:value for key,value in raw.items() if key in asdict(base)}
    opsd={key:value for key,value in raw.items() if key not in asdict(base)}
    return load_settings(base_file,extra),opsd


def validate_opsd_config(cfg,opsd):
    defaults={'mode':'recovery','training_partitions':['rl'],'collection_rollouts_per_task':4,'max_candidates':48,'preservation_actions_per_task':4,'hint_temperature':0.0,
        'hint_max_tokens':384,'recovery_max_actions':12,'recovery_weight':1.0,
        'max_hints_per_instance':3,
        'preservation_weight':0.25,'epochs_per_round':2,'eval_rollouts_per_task':1,
        'gradient_accumulation_steps':4,'eval_steps':50,'save_steps':50,'save_total_limit':2,
        'output_dir':cfg.output_dir}
    unknown=set(opsd)-set(defaults)
    if unknown: raise ValueError('Unknown OPSD settings: '+', '.join(sorted(unknown)))
    defaults.update(opsd)
    if not isinstance(defaults['training_partitions'],list) or not defaults['training_partitions'] or any(x not in ('rl','sft') for x in defaults['training_partitions']): raise ValueError('training_partitions must contain only rl and/or sft')
    if defaults['mode'] not in ('recovery','hint'): raise ValueError('mode must be recovery or hint')
    for key in ('collection_rollouts_per_task','max_candidates','preservation_actions_per_task',
        'hint_max_tokens','recovery_max_actions','max_hints_per_instance','epochs_per_round','eval_rollouts_per_task',
        'gradient_accumulation_steps','eval_steps','save_steps','save_total_limit'):
        if type(defaults[key]) is not int or defaults[key]<1: raise ValueError(key+' must be a positive integer')
    for key in ('hint_temperature','recovery_weight','preservation_weight'):
        if type(defaults[key]) not in (int,float) or not math.isfinite(defaults[key]) or defaults[key]<0: raise ValueError(key+' must be nonnegative')
    if defaults['hint_temperature']>2: raise ValueError('hint_temperature must be <= 2')
    if defaults['recovery_weight']+defaults['preservation_weight']<=0: raise ValueError('At least one loss weight must be positive')
    defaults['output_dir']=str(path(defaults['output_dir']))
    return defaults


def training_task_ids(cfg,opsd,partition='rl'):
    if partition!='rl': return task_ids(cfg,partition)
    return list(dict.fromkeys(t for pool in opsd['training_partitions'] for t in task_ids(cfg,pool)))


def tool_error(message):
    if message.get('role')!='tool': return False
    if message.get('error') is True or message.get('name')=='__runtime__': return True
    content=message.get('content')
    if isinstance(content,str):
        try: content=json.loads(content)
        except (TypeError,ValueError): return False
    if isinstance(content,dict):
        body=content.get('content',content)
        if isinstance(body,dict):
            return bool(body.get('error')) or str(body.get('status','')).lower() in {'error','failed','failure'} or body.get('success') is False
        if isinstance(body,str) and re.match(r'^\s*(?:error|failed)\s*:',body,re.I): return True
    if isinstance(content,str) and re.match(r'^\s*(?:error|failed)\s*:',content,re.I): return True
    return False


def find_candidates(episode,source_file):
    if not episode.get('complete') or episode.get('reward') is None:
        return []
    actions=episode.get('actions') or [];messages=episode.get('messages') or [];found=[]
    for index,action in enumerate(actions[:-1]):
        start=int(action['raw_message_count'])+1
        end=int(actions[index+1]['raw_message_count'])
        observations=messages[start:end]
        errors=[(start+offset,m) for offset,m in enumerate(observations) if tool_error(m)]
        if not errors: continue
        error_index,error_message=errors[0]
        if not any(m.get('role') in ('tool','user') for m in observations): continue
        found.append(dict(candidate_id=f"{episode['episode_id']}:after-turn-{action['turn']}",
            source_file=str(source_file),episode_id=episode['episode_id'],task_id=str(episode['task_id']),
            seed=episode['seed'],baseline_reward=episode.get('reward'),failure_turn=action['turn'],
            recovery_turn=actions[index+1]['turn'],failure_action=action.get('text',''),
            error_message=copy.deepcopy(error_message),error_message_index=error_index,
            recovery_messages=copy.deepcopy(messages[:end]),full_messages=copy.deepcopy(messages),
            policy_version=episode.get('policy_version'),classification='explicit_tool_error',
            replay_verified=False,hint=None,hinted_reward=None,hint_validation_passed=False))
    return found


def _canonical(messages):
    # Tau2 may assign fresh internal message IDs on a reset. Those are transport
    # metadata, not model-visible environment changes; prompt IDs are checked
    # separately before accepting a replay.
    keep=('role','content','name','error')
    return [{k:m[k] for k in keep if k in m and m[k] is not None} for m in messages]


def adapter_fingerprint(directory):
    directory=Path(directory)
    files=sorted(directory.glob('adapter_model.*'))
    if not files: raise FileNotFoundError('No adapter weights under '+str(directory))
    h=hashlib.sha256()
    for file in files:
        h.update(file.name.encode());h.update(file.read_bytes())
    return h.hexdigest()


def create_runtime(policy,cfg,seed):
    from openai import OpenAI
    from transformers import AutoTokenizer
    from tau2_grpo.rl_context import ContextController,FrozenSummarizer
    from tau2_grpo.rl_tau2 import ExternalChat,RetailEnvironment
    ext_tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
    client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=180,max_retries=0)
    external=ExternalChat(client,ext_tokenizer,cfg,[])
    summarizer=FrozenSummarizer(client,cfg.user_model,ext_tokenizer,cfg.external_context_limit,cfg.summary_max_tokens)
    controller=ContextController(policy.tokenizer,summarizer,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
    env=RetailEnvironment(seed=seed,task_id='',cfg=cfg,external=external)
    return client,controller,env


def replay_to_recovery(policy,cfg,candidate):
    """Reconstruct the state and reject any prompt/observation replay mismatch."""
    from tau2_grpo.rl_tau2 import RetailEnvironment
    client,controller,env=create_runtime(policy,cfg,int(candidate['seed']))
    try:
        env.task_id=str(candidate['task_id']);raw=env.reset()
        episode=json.loads(Path(candidate['source_file']).read_text())
        actions=episode['actions'];recovery_turn=int(candidate['recovery_turn'])
        if not 0<recovery_turn<len(actions): raise ValueError('Recovery point lacks a following recorded action')
        for index,action in enumerate(actions[:recovery_turn]):
            prepared=controller.prepare(raw,int(candidate['seed'])+int(action['turn']))
            if prepared['prompt_ids']!=action['prompt_ids']:
                raise RuntimeError(f"prompt_replay_mismatch_at_turn_{action['turn']}")
            raw_count=int(action['raw_message_count'])
            assistant=episode['messages'][raw_count]
            if assistant.get('role')!='assistant' or assistant.get('content')!=action.get('text'):
                raise RuntimeError('saved assistant turn does not align with action record')
            raw.append(copy.deepcopy(assistant))
            incoming,done,_call=env.step(action['text'])
            expected_end=int(actions[index+1]['raw_message_count'])
            expected=episode['messages'][raw_count+1:expected_end]
            if _canonical(incoming)!=_canonical(expected):
                raise RuntimeError(f"environment_replay_mismatch_after_turn_{action['turn']}")
            raw.extend(incoming)
            if done: raise RuntimeError('environment became terminal before recovery point')
        recovery_action=actions[recovery_turn]
        prepared=controller.prepare(raw,int(candidate['seed'])+recovery_turn)
        if prepared['prompt_ids']!=recovery_action['prompt_ids']:
            raise RuntimeError(f"recovery_prompt_mismatch_at_turn_{recovery_turn}")
        return client,controller,env,raw,prepared
    except Exception:
        try: env.close()
        finally: client.close()
        raise


def _token_ids(tokenizer,messages):
    return tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,
        add_generation_prompt=True,enable_thinking=False)


def diagnostic_messages(candidate,tokenizer,cfg,opsd):
    from tau2_grpo.rl_context import student_messages
    system=(ROOT/'templates/retail_opsd_diagnoser.txt').read_text()
    prefix=student_messages(candidate['recovery_messages'])
    full=student_messages(candidate['full_messages'])
    retry_number=int(candidate.get('retry_number',1))
    instruction=('Diagnose the observed failure after the rollout has ended. Return a short safe next-turn hint grounded only in recovery_state.')
    if retry_number>1:
        instruction=('The previous hinted recovery attempt did not solve the task. Diagnose what still went wrong in the hindsight trajectory, then provide a revised next-turn hint to append at the original recovery_state. The agent will restart from that exact state. Ground the hint only in information visible in recovery_state; never reveal facts learned later.')
    payload={'task_id':candidate['task_id'],'failure_turn':candidate['failure_turn'],
        'retry_number':retry_number,
        'previous_hints':candidate.get('previous_hints',[]),
        'previous_attempt_reward':candidate.get('previous_attempt_reward'),
        'recovery_state':prefix,'completed_student_trajectory_for_hindsight':full,
        'observed_tool_error':candidate['error_message'],
        'final_tau2_reward':candidate.get('baseline_reward'),
        'instruction':instruction}
    messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
    if len(_token_ids(tokenizer,messages))+opsd['hint_max_tokens']>cfg.external_context_limit:
        # Preserve the entire causal prefix and shrink only hindsight context.
        payload['completed_student_trajectory_for_hindsight']=full[:2]+full[-12:]
        payload['hindsight_excerpted']=True
        messages[1]['content']=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if len(_token_ids(tokenizer,messages))+opsd['hint_max_tokens']>cfg.external_context_limit:
        raise ValueError('Hint diagnosis input exceeds teacher context even after excerpting hindsight')
    return messages


def validate_annotation(raw):
    try: item=json.loads(raw)
    except (TypeError,ValueError) as error: raise ValueError('diagnosis is not JSON') from error
    keys={'diagnosis','confidence','recoverability','hint'}
    if not isinstance(item,dict) or set(item)!=keys: raise ValueError('diagnosis schema must contain exactly '+str(sorted(keys)))
    if item['confidence'] not in ('high','medium','low'): raise ValueError('invalid confidence')
    if item['recoverability'] not in ('recoverable','uncertain','not_recoverable'): raise ValueError('invalid recoverability')
    for name,lo,hi in (('diagnosis',4,80),('hint',4,40)):
        value=item[name]
        if not isinstance(value,str) or not lo<=len(value.split())<=hi: raise ValueError(name+' has invalid length')
        if re.search(r'<tool_call>|```|\b(?:ignore|override) (?:the )?(?:system|policy)\b',value,re.I): raise ValueError(name+' contains unsafe control text')
    if re.search(r'\b(?:the answer is|the correct value is|use this exact value)\b',item['hint'],re.I):
        raise ValueError('hint appears to disclose a target value')
    return item


def reject_hint_leakage(annotation,candidate):
    prefix=' '.join(str(m.get('content','')) for m in candidate['recovery_messages'])
    future=' '.join(str(m.get('content','')) for m in candidate['full_messages'][len(candidate['recovery_messages']):])
    patterns=(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
        r'(?<![A-Z0-9])#?[A-Z]{1,4}\d{5,}(?![A-Z0-9])',r'(?<!\d)\d{5}(?!\d)')
    for pattern in patterns:
        for value in re.findall(pattern,future,re.I):
            if value not in prefix and value.lower() in annotation['hint'].lower():
                raise ValueError('hint contains a literal found only after the recovery point')


def request_diagnosis(client,tokenizer,candidate,cfg,opsd):
    messages=diagnostic_messages(candidate,tokenizer,cfg,opsd)
    attempts=[];annotation=None;raw='';last_error=None
    for attempt in range(3):
        response=client.chat.completions.create(model=cfg.user_model,messages=messages,
            temperature=opsd['hint_temperature'],max_tokens=opsd['hint_max_tokens'],
            response_format={'type':'json_object'},
            extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        raw=response.choices[0].message.content or ''
        if response.choices[0].finish_reason=='length':
            last_error='Hint diagnosis was truncated'
        else:
            try:
                annotation=validate_annotation(raw)
                reject_hint_leakage(annotation,candidate)
                last_error=None
                break
            except ValueError as error:
                last_error=str(error)
        attempts.append({'attempt':attempt,'response':raw,'error':last_error})
        if attempt<2:
            messages.extend([{'role':'assistant','content':raw},
                {'role':'user','content':'Return a much shorter annotation that passes the schema. Keep diagnosis to 4-60 words and hint to 4-30 words. Do not summarize the full conversation; state the cause and one safe next action. Output only the required JSON object.'}])
    return annotation,dict(error=last_error,attempts=attempts,raw=raw,messages=messages,
        hindsight_excerpted=bool(json.loads(messages[1]['content']).get('hindsight_excerpted')))


def generate_hints(run,cfg,opsd):
    from openai import OpenAI
    from transformers import AutoTokenizer
    run=Path(run);candidates=[]
    for file in sorted((run/'collection').glob('episode-*.json')):
        episode=json.loads(file.read_text());candidates.extend(find_candidates(episode,file))
    candidates=candidates[:opsd['max_candidates']]
    tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
    client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=180,max_retries=0)
    saved=[]
    try:
        for candidate in candidates:
            annotation,diagnosis=request_diagnosis(client,tokenizer,candidate,cfg,opsd)
            candidate['diagnosis_annotation']=annotation
            candidate['diagnosis_error']=diagnosis['error']
            candidate['diagnosis_attempts']=diagnosis['attempts']
            candidate['diagnosis_raw']=diagnosis['raw']
            candidate['diagnosis_model']=cfg.user_model
            candidate['diagnosis_messages']=diagnosis['messages']
            candidate['hindsight_excerpted']=diagnosis['hindsight_excerpted']
            saved.append(candidate)
            atomic_json(run/'candidates.json',saved)
            print(json.dumps({'candidate_id':candidate['candidate_id'],'task_id':candidate['task_id'],
                'failure_turn':candidate['failure_turn'],
                'recoverability':annotation['recoverability'] if annotation else None,
                'hint':annotation['hint'] if annotation else None,'error':diagnosis['error']},ensure_ascii=False),flush=True)
    finally: client.close()
    (run/'hints.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in saved))
    return saved


def rollout_from_recovery(policy,cfg,candidate,with_hint=False,max_actions=12,
                          hint_tokenizer=None,max_hints_per_instance=1,opsd=None):
    hint=candidate['diagnosis_annotation']['hint'] if with_hint else None
    client,controller,env,raw,prepared=replay_to_recovery(policy,cfg,candidate)
    actions=[];done=False;total_tokens=0;error=None;simulation=None;reward=None
    hint_events=([{'turn':int(candidate['recovery_turn']),'hint':hint,
        'diagnosis':candidate['diagnosis_annotation'].get('diagnosis'),
        'confidence':candidate['diagnosis_annotation'].get('confidence')}]
        if hint else [])
    try:
        for offset in range(max_actions):
            guidance=hint
            hint=None
            prepared=controller.prepare(raw,int(candidate['seed'])+int(candidate['recovery_turn'])+offset,
                guidance=guidance)
            result=policy.generate(prepared,cfg.response_tokens,
                int(candidate['seed'])+int(candidate['recovery_turn'])+offset)
            if not result['completion_ids'] or not result['ended_with_eos']:
                raise RuntimeError('recovery_action_truncated_or_empty')
            action=dict(**prepared,**result,turn=int(candidate['recovery_turn'])+offset,
                policy_version=policy.version,raw_message_count=len(raw),temperature=cfg.temperature)
            actions.append(action);total_tokens+=len(result['completion_ids'])
            raw.append({'role':'assistant','content':result['text']})
            incoming,done,_call=env.step(result['text']);raw.extend(incoming)
            if done: break
        reward,simulation=env.finish()
        return dict(complete=True,reward=reward,simulation=simulation,messages=raw,actions=actions,
            summaries=copy.deepcopy(controller.summaries),external=copy.deepcopy(env.external.audit),
            total_tokens=total_tokens,used_hint=with_hint,hint_events=hint_events,
            stop_reason='terminal' if done else 'recovery_action_budget')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        return dict(complete=False,reward=None,error=error,messages=raw,actions=actions,
            summaries=copy.deepcopy(controller.summaries),external=copy.deepcopy(env.external.audit),
            total_tokens=total_tokens,used_hint=with_hint,hint_events=hint_events)
    finally:
        try: env.close()
        finally: client.close()


def validate_candidates(policy,cfg,run,candidates,opsd):
    from openai import OpenAI
    from transformers import AutoTokenizer
    hint_tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
    retry_client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=180,max_retries=0)
    output=[]
    for candidate in candidates:
        ann=candidate.get('diagnosis_annotation') or {}
        if ann.get('recoverability')!='recoverable':
            candidate.update(replay_verified=False,hint_validation_passed=False,
                validation_reason='diagnoser_did_not_mark_recoverable')
            output.append(candidate)
            atomic_json(Path(run)/'validation.json',output)
            print(json.dumps({'candidate_id':candidate['candidate_id'],'replay_verified':False,
                'hint_validation_passed':False,'reason':candidate['validation_reason']}),flush=True)
            continue
        try:
            # Save an unhinted on-policy action at exactly the same replayed state.
            client,controller,env,raw,student_prompt=replay_to_recovery(policy,cfg,candidate)
            try:
                student_action=policy.generate(student_prompt,cfg.response_tokens,
                    int(candidate['seed'])+int(candidate['recovery_turn']))
                candidate['student_prompt_ids']=student_prompt['prompt_ids']
                candidate['student_completion_ids']=student_action['completion_ids']
            finally:
                env.close();client.close()
            unhinted=rollout_from_recovery(policy,cfg,candidate,False,opsd['recovery_max_actions'])
            candidate['unhinted_reward']=unhinted.get('reward');candidate['unhinted_rollout']=unhinted
            candidate['replay_verified']=True
            first_unhinted=unhinted.get('actions',[{}])[0].get('completion_ids')
            if first_unhinted!=candidate['student_completion_ids']:
                raise RuntimeError('saved student sample differs from executed unhinted continuation')
            hint_attempts=[];current_annotation=ann;successful_attempt=None;last_hint_rollout=None
            for attempt_index in range(opsd['max_hints_per_instance']):
                attempt_number=attempt_index+1
                attempt_candidate=dict(candidate,diagnosis_annotation=current_annotation)
                hinted=rollout_from_recovery(policy,cfg,attempt_candidate,True,opsd['recovery_max_actions'])
                attempt_record={'attempt':attempt_number,'hint':current_annotation['hint'],
                    'diagnosis':current_annotation,'reward':hinted.get('reward'),
                    'complete':hinted.get('complete'),'rollout':hinted}
                hint_attempts.append(attempt_record);last_hint_rollout=hinted
                if hinted.get('complete') and hinted.get('reward')==1.0:
                    successful_attempt=attempt_record
                    break
                # Retry only a completed, scored failure. Infrastructure or
                # truncation failures do not provide a reliable diagnosis target.
                if not hinted.get('complete') or attempt_index+1>=opsd['max_hints_per_instance']:
                    break
                later_errors=[m for m in hinted.get('messages',[]) if tool_error(m)]
                diagnostic_state=dict(candidate,
                    candidate_id=f"{candidate['candidate_id']}:retry-{attempt_number+1}",
                    recovery_messages=copy.deepcopy(candidate['recovery_messages']),
                    full_messages=copy.deepcopy(hinted['messages']),
                    error_message=copy.deepcopy(later_errors[-1] if later_errors else candidate['error_message']),
                    baseline_reward=hinted.get('reward'),retry_number=attempt_number+1,
                    previous_hints=[x['hint'] for x in hint_attempts],
                    previous_attempt_reward=hinted.get('reward'))
                retry_annotation,retry_diagnosis=request_diagnosis(retry_client,hint_tokenizer,
                    diagnostic_state,cfg,opsd)
                attempt_record['next_diagnosis']={'annotation':retry_annotation,
                    'error':retry_diagnosis['error'],'attempts':retry_diagnosis['attempts'],
                    'messages':retry_diagnosis['messages']}
                if not retry_annotation or retry_annotation.get('recoverability')!='recoverable':
                    break
                current_annotation=retry_annotation
            candidate['hint_attempts']=hint_attempts
            hinted=successful_attempt['rollout'] if successful_attempt else last_hint_rollout
            candidate['hinted_reward']=hinted.get('reward') if hinted else None
            candidate['hinted_rollout']=hinted
            candidate['successful_hint_attempt']=successful_attempt['attempt'] if successful_attempt else None
            candidate['hint_validation_passed']=bool(successful_attempt and unhinted.get('complete')
                and unhinted.get('reward') is not None and unhinted.get('reward')<1.0)
            if successful_attempt:
                chosen_hint=successful_attempt['hint']
                client,controller,env,raw,_prompt=replay_to_recovery(policy,cfg,candidate)
                try:
                    teacher_prompt=controller.prepare(raw,int(candidate['seed'])+int(candidate['recovery_turn']),
                        guidance=chosen_hint)
                    candidate['teacher_prompt_ids']=teacher_prompt['prompt_ids']
                finally:
                    env.close();client.close()
                candidate['distillation_examples']=[{'turn':int(candidate['recovery_turn']),
                    'hint':chosen_hint,'student_prompt_ids':candidate['student_prompt_ids'],
                    'teacher_prompt_ids':candidate['teacher_prompt_ids'],
                    'student_completion_ids':candidate['student_completion_ids']}]
            else:
                candidate['distillation_examples']=[]
            if candidate['hint_validation_passed']:
                candidate['validation_reason']=f"hint_attempt_{successful_attempt['attempt']}_reached_reward_1_and_unhinted_below_1"
            elif not unhinted.get('complete'):
                candidate['validation_reason']='unhinted_replay_failed: '+str(unhinted.get('error'))
            else:
                candidate['validation_reason']=f"no_reward_1_recovery_after_{len(hint_attempts)}_hint_attempts"
        except Exception as exc:
            candidate.update(replay_verified=False,hint_validation_passed=False,
                validation_reason=f'{type(exc).__name__}: {exc}')
        output.append(candidate)
        atomic_json(Path(run)/'validation.json',output)
        print(json.dumps({'candidate_id':candidate['candidate_id'],'replay_verified':candidate['replay_verified'],
            'hint_validation_passed':candidate['hint_validation_passed'],'hinted_reward':candidate.get('hinted_reward'),
            'reason':candidate['validation_reason']},ensure_ascii=False),flush=True)
    retry_client.close()
    (Path(run)/'verified_recovery.jsonl').write_text(''.join(
        json.dumps({'candidate_id':c['candidate_id'],'task_id':c['task_id'],'seed':c['seed'],
            'failure_turn':c['failure_turn'],'recovery_turn':c['recovery_turn'],
            'hint':c.get('diagnosis_annotation',{}).get('hint'),
            'student_prompt_ids':c.get('student_prompt_ids'),'teacher_prompt_ids':c.get('teacher_prompt_ids'),
            'student_completion_ids':c.get('student_completion_ids'),'verified':c.get('hint_validation_passed'),
            'distillation_examples':c.get('distillation_examples',[]),
            'successful_hint_attempt':c.get('successful_hint_attempt'),
            'hinted_reward':c.get('hinted_reward'),'validation_reason':c.get('validation_reason')},ensure_ascii=False)+'\n'
        for c in output if c.get('hint_validation_passed')))
    return output


def collect(policy,cfg,run,opsd,partition='rl',selected_task_ids=None,resume=False):
    from tau2_grpo.rl import make_episode
    import wandb
    available=training_task_ids(cfg,opsd,partition)
    ids=list(selected_task_ids) if selected_task_ids else available
    if len(ids)!=len(set(ids)) or not set(ids).issubset(available):
        raise ValueError('Selected collection task IDs must be unique members of the requested Tau2 split')
    n=opsd['collection_rollouts_per_task']
    records=[]
    manifest={'partition':partition,'training_partitions':opsd['training_partitions'],'task_ids':ids,'rollouts_per_task':n,
        'task_split_manifest_sha256':hashlib.sha256(path(cfg.split_manifest).read_bytes()).hexdigest(),
        'initial_adapter':str(path(cfg.initial_adapter)),
        'initial_adapter_sha256':adapter_fingerprint(path(cfg.initial_adapter)),
        'training_started':False}
    manifest_path=Path(run)/'collection-manifest.json'
    if resume and manifest_path.is_file():
        previous=json.loads(manifest_path.read_text())
        for key in ('partition','task_ids','rollouts_per_task','task_split_manifest_sha256','initial_adapter_sha256'):
            if previous.get(key)!=manifest.get(key):
                raise RuntimeError('Cannot resume collection: manifest mismatch for '+key)
    atomic_json(manifest_path,manifest)
    wb=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=Path(run).name+'-collect',
        dir=str(run),config=asdict(cfg)|opsd|manifest)
    if wb is None or wb.disabled: raise RuntimeError('W&B collection initialization failed')
    table=wandb.Table(columns=['task_id','seed','reward','complete','actor_turns','tool_responses',
        'stop_reason','error','conversation','checks_json'])
    try:
        for task_index,task_id in enumerate(ids):
            for rollout in range(n):
                seed=cfg.seed+task_index*n+rollout
                out=Path(run)/'collection'/f'episode-{task_id}-{seed}.json'
                if resume and out.is_file():
                    try: existing=json.loads(out.read_text())
                    except (OSError,ValueError): existing={}
                    if existing.get('complete') is True and existing.get('reward') is not None:
                        records.append(dict(task_id=task_id,seed=seed,complete=True,
                            reward=existing['reward'],file=str(out),resumed=True))
                        print(json.dumps({'event':'resume_skip_complete','task_id':task_id,'seed':seed,
                            'reward':existing['reward'],'file':str(out)}),flush=True)
                        continue
                episode=make_episode(policy,cfg,task_id,seed,out,with_reference=False)
                from tau2_grpo.rl_reporting import episode_check_report
                report=episode_check_report(episode)
                records.append(dict(task_id=task_id,seed=seed,complete=episode.get('complete'),reward=episode.get('reward'),file=str(out)))
                atomic_json(Path(run)/'collection-progress.json',records)
                conversation='\n\n'.join(str(m.get('role','')).upper()+': '+str(m.get('content',''))
                    for m in episode.get('messages',[]))
                table.add_data(str(task_id),seed,episode.get('reward'),episode.get('complete'),
                    report['turn_counts']['actor_turns'],report['turn_counts']['tool_response_messages'],
                    episode.get('stop_reason'),episode.get('error'),conversation,
                    json.dumps(report['checks'],ensure_ascii=False))
                metrics={'collection/reward':episode.get('reward'),
                    'collection/complete':episode.get('complete'),'collection/task_id':int(task_id)}
                if len(records)%4==0 or len(records)==len(ids)*n:
                    metrics['collection/episode_samples']=table
                wb.log(metrics)
                print(json.dumps({'event':'collected','task_id':task_id,'seed':seed,
                    'complete':episode.get('complete'),'reward':episode.get('reward'),'file':str(out)}),flush=True)
    finally: wb.finish()
    return records


def evaluate(cfg,run,opsd,stage,checkpoint=None,selected_task_ids=None):
    from dataclasses import replace
    from tau2_grpo.rl import make_episode
    from tau2_grpo.rl_policy import LocalPolicy
    from tau2_grpo.rl_reporting import episode_check_report
    import wandb
    if stage=='before': adapter=path(cfg.initial_adapter)
    else:
        if checkpoint and checkpoint!='latest': adapter=path(checkpoint)
        else:
            options=[]
            for item in Path(run).glob('checkpoint-*'):
                match=re.fullmatch(r'checkpoint-(\d+)',item.name)
                if match:
                    adapter_path=item if (item/'adapter_config.json').is_file() else item/'policy'
                    if (adapter_path/'adapter_config.json').is_file(): options.append((int(match.group(1)),adapter_path))
            if not options: raise FileNotFoundError('No trained OPSD adapter checkpoint to evaluate')
            adapter=max(options,key=lambda item:item[0])[1]
    eval_cfg=replace(cfg,initial_adapter=str(adapter))
    policy=LocalPolicy(eval_cfg,adapter=True,trainable=False)
    available=task_ids(cfg,'validation')
    ids=list(selected_task_ids) if selected_task_ids else available
    if len(ids)!=len(set(ids)) or not set(ids).issubset(available):
        raise ValueError('Evaluation task IDs must be unique members of Tau2 validation')
    rows=[];outdir=Path(run)/'evaluation'/stage
    outdir.mkdir(parents=True,exist_ok=True)
    wb=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,
        name=Path(run).name+'-eval-'+stage,dir=str(outdir),
        config=asdict(cfg)|opsd|{'stage':stage,'adapter':str(adapter),'partition':'validation','task_ids':ids})
    if wb is None or wb.disabled: raise RuntimeError('W&B evaluation initialization failed')
    table=wandb.Table(columns=['task_id','seed','reward','complete','actor_turns','tool_calls','check_counts','conversation','checks_json'])
    try:
        for task_index,task_id in enumerate(ids):
            for sample in range(opsd['eval_rollouts_per_task']):
                seed=cfg.seed+100000+task_index*opsd['eval_rollouts_per_task']+sample
                episode_file=outdir/f'task-{task_id}-seed-{seed}.json'
                episode=make_episode(policy,eval_cfg,task_id,seed,episode_file,with_reference=False)
                report=episode_check_report(episode)
                row={'task_id':str(task_id),'seed':seed,'reward':episode.get('reward'),
                    'complete':episode.get('complete'),'checks':report['check_counts'],
                    'turn_counts':report['turn_counts'],'error':episode.get('error'),
                    'episode_file':str(episode_file)}
                rows.append(row)
                conversation='\n\n'.join(str(m.get('role','')).upper()+': '+str(m.get('content',''))
                    for m in episode.get('messages',[]))
                table.add_data(str(task_id),seed,episode.get('reward'),episode.get('complete'),
                    report['turn_counts']['actor_turns'],report['turn_counts']['tool_response_messages'],
                    json.dumps(report['check_counts']),conversation,json.dumps(report['checks'],ensure_ascii=False))
                wb.log({'eval/'+stage+'/reward':episode.get('reward'),'eval/'+stage+'/task_id':int(task_id),
                    'eval/'+stage+'/actor_turns':report['turn_counts']['actor_turns']})
                print(json.dumps(row,ensure_ascii=False),flush=True)
        evaluated=[r for r in rows if r['complete'] and r['reward'] is not None]
        state_file=Path(run)/'evaluation'/'validation_recovery_states.json'
        if stage=='before':
            candidates=[]
            for row in rows:
                if row['complete']:
                    episode=json.loads(Path(row['episode_file']).read_text())
                    candidates.extend(find_candidates(episode,row['episode_file']))
            candidates=candidates[:opsd['max_candidates']]
            atomic_json(state_file,candidates)
        else:
            candidates=json.loads(state_file.read_text()) if state_file.exists() else []
        recovery_rows=[]
        recovery_table=wandb.Table(columns=['candidate_id','task_id','failure_turn','reward',
            'complete','actor_turns','error','conversation'])
        for candidate in candidates:
            try:
                result=rollout_from_recovery(policy,cfg,candidate,False,opsd['recovery_max_actions'])
            except Exception as exc:
                result={'complete':False,'reward':None,'error':f'{type(exc).__name__}: {exc}',
                    'actions':[],'messages':[]}
            item={'candidate_id':candidate['candidate_id'],'task_id':candidate['task_id'],
                'failure_turn':candidate['failure_turn'],'complete':result.get('complete'),
                'reward':result.get('reward'),'error':result.get('error'),
                'actor_turns':len(result.get('actions',[])),'conversation':result.get('messages',[])}
            recovery_rows.append(item)
            conversation='\n\n'.join(str(m.get('role','')).upper()+': '+str(m.get('content',''))
                for m in item['conversation'])
            recovery_table.add_data(item['candidate_id'],item['task_id'],item['failure_turn'],
                item['reward'],item['complete'],item['actor_turns'],item['error'],conversation)
            wb.log({'eval/'+stage+'/recovery_reward':item['reward'],
                'eval/'+stage+'/recovery_turns':item['actor_turns']})
            print(json.dumps({'event':'heldout_recovery','stage':stage,'candidate_id':item['candidate_id'],
                'reward':item['reward'],'complete':item['complete'],'error':item['error']}),flush=True)
        atomic_json(Path(run)/'evaluation'/f'recovery-{stage}.json',recovery_rows)
        recovery_eval=[item for item in recovery_rows if item['complete'] and item['reward'] is not None]
        recovery_summary={'states':len(candidates),'evaluated':len(recovery_eval),
            'successes':sum(item['reward']==1.0 for item in recovery_eval),
            'mean_reward':sum(item['reward'] for item in recovery_eval)/len(recovery_eval) if recovery_eval else None,
            'scope':'Only explicit, nonterminal tool errors with a subsequent recorded agent turn.'}
        summary={'stage':stage,'adapter':str(adapter),'partition':'validation','tasks':len(ids),
            'episodes':len(rows),'evaluated':len(evaluated),'infrastructure_errors':len(rows)-len(evaluated),
            'successes':sum(r['reward']==1.0 for r in evaluated),
            'mean_reward':sum(r['reward'] for r in evaluated)/len(evaluated) if evaluated else None,
            'results':rows,'conditional_recovery':recovery_summary,
            'note':'Fixed Tau2 validation tasks; no hints. One or more rollouts per task as configured.'}
        atomic_json(outdir/'summary.json',summary)
        wb.log({'eval/'+stage+'/summary':table,'eval/'+stage+'/mean_reward':summary['mean_reward'],
            'eval/'+stage+'/recovery_summary':recovery_summary,
            'eval/'+stage+'/recovery_samples':recovery_table,
            'eval/'+stage+'/pass_rate':summary['successes']/len(evaluated) if evaluated else None})
        return summary
    finally: wb.finish()


def preserved_examples(run,per_task=4,seed=42):
    grouped={}
    for file in sorted((Path(run)/'collection').glob('episode-*.json')):
        episode=json.loads(file.read_text())
        if episode.get('complete') and episode.get('reward')==1.0:
            for action in episode.get('actions',[]):
                if action.get('completion_ids') and action.get('prompt_ids'):
                    grouped.setdefault(str(episode['task_id']),[]).append(
                        {'task_id':str(episode['task_id']),'seed':episode['seed'],
                         'prompt_ids':action['prompt_ids'],'completion_ids':action['completion_ids']})
    rng=random.Random(seed);examples=[]
    for task_id in sorted(grouped):
        items=grouped[task_id];rng.shuffle(items);examples.extend(items[:per_task])
    return examples


def train(policy,cfg,run,opsd):
    import torch
    import wandb
    from tau2_grpo.hint_distillation import turn_distillation_loss
    from tau2_grpo.aligntune_runtime import make_owner
    hint_mode=opsd['mode']=='hint'
    data_path=Path(run)/('hint_turns.jsonl' if hint_mode else 'verified_recovery.jsonl')
    recovery=[json.loads(x) for x in data_path.read_text().splitlines() if x.strip()] if data_path.exists() else []
    preserve=[] if hint_mode else preserved_examples(run,opsd['preservation_actions_per_task'],cfg.seed)
    if not recovery: raise RuntimeError('No prepared training examples for mode '+opsd['mode'])
    manifest_file=Path(run)/'collection-manifest.json'
    if not manifest_file.is_file(): raise RuntimeError('Missing collection manifest')
    manifest=json.loads(manifest_file.read_text())
    allowed=set(training_task_ids(cfg,opsd))
    if manifest.get('partition')!='rl' or not set(manifest.get('task_ids',[])).issubset(allowed):
        raise RuntimeError('OPSD optimizer accepts only configured retail training partitions')
    if any(str(item.get('task_id')) not in allowed for item in recovery):
        raise RuntimeError('A recovery example comes from outside the Tau2 retail RL split')
    source_hash=adapter_fingerprint(path(cfg.initial_adapter))
    if manifest.get('initial_adapter_sha256')!=source_hash:
        raise RuntimeError('Collection adapter differs from the configured frozen teacher/student checkpoint')
    if policy.has_reference: policy.sync_reference()
    optimizer=torch.optim.AdamW((p for p in policy.model.parameters() if p.requires_grad),lr=cfg.learning_rate)
    wandb_run=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=Path(run).name,
        dir=str(run),config=asdict(cfg)|opsd|{'verified_recovery_examples':len(recovery),'preservation_actions':len(preserve)})
    if wandb_run is None or wandb_run.disabled: raise RuntimeError('W&B initialization failed')
    if hint_mode:
        hint_samples=wandb.Table(columns=['episode_id','task_id','turn','reward','diagnosis','original_action','teacher_feedback'])
        for item in recovery:
            hint_samples.add_data(item['episode_id'],item['task_id'],item['turn'],item['reward'],item['diagnosis'],item['original_action'],item['hint'])
        wandb_run.log({'hint/turn_samples':hint_samples})
    candidate_file=Path(run)/'validation.json'
    candidates=json.loads(candidate_file.read_text()) if candidate_file.exists() and not hint_mode else []
    samples=wandb.Table(columns=['candidate_id','task_id','baseline_reward','failure_turn',
        'diagnosis','confidence','hint','recoverability','replay_verified','hint_validated',
        'unhinted_recovery_reward','hinted_reward','validation_reason','hint_sequence',
        'diagnosis_events','hinted_conversation'])
    for item in candidates:
        ann=item.get('diagnosis_annotation') or {};rollout=item.get('hinted_rollout') or {}
        conversation='\n\n'.join(str(m.get('role','')).upper()+': '+str(m.get('content',''))
            for m in rollout.get('messages',[]))
        samples.add_data(item.get('candidate_id'),item.get('task_id'),item.get('baseline_reward'),
            item.get('failure_turn'),ann.get('diagnosis'),ann.get('confidence'),ann.get('hint'),
            ann.get('recoverability'),item.get('replay_verified'),item.get('hint_validation_passed'),
            item.get('unhinted_reward'),item.get('hinted_reward'),item.get('validation_reason'),
            json.dumps([{'attempt':x.get('attempt'),'hint':x.get('hint'),'reward':x.get('reward')}
                for x in item.get('hint_attempts',[])],ensure_ascii=False),
            json.dumps([x.get('next_diagnosis') for x in item.get('hint_attempts',[])
                if x.get('next_diagnosis')],ensure_ascii=False),conversation)
    wandb_run.log({'opsd/intervention_samples':samples})
    updates=0;history=[];owner=None
    def execute():
        nonlocal updates
        from tau2_grpo.retail_distillation import fit
        result=fit(policy,optimizer,cfg,opsd,run,recovery,preserve,wandb_run,
            on_step=lambda step,metrics: owner.step(step,metrics) if owner is not None else None)
        updates=result['global_step']
        return result
    manifest['training_started']=True;atomic_json(manifest_file,manifest)
    try:
        owner=make_owner(kind='tau2_retail_opsd_privileged_distillation',config=cfg,
            run_dir=Path(run),model=policy.model,dataset={'recovery':recovery,'preservation':preserve},
            execute=execute,tokenizer=policy.tokenizer)
        owner.train()
        atomic_json(Path(run)/'result.json',{'trained':updates>0,'optimizer_steps':updates,
            'mode':opsd['mode'],'training_examples':len(recovery),'preservation_actions':len(preserve),
            'recovery_hint_turn_examples':sum(len(x.get('distillation_examples') or [x]) for x in recovery),
            'max_hints_per_instance':opsd['max_hints_per_instance'],
            'loss':'top-100-plus-tail symmetric JSD over student-sampled action tokens',
            'hint_free_student_prompts':True})
    finally: wandb_run.finish()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','collect','diagnose','validate','prepare-hint','train','evaluate'))
    parser.add_argument('--config',default=str(ROOT/'configs/opsd-retail.json'))
    parser.add_argument('--run-dir')
    parser.add_argument('--partition',choices=('rl','validation'),default='rl')
    parser.add_argument('--task-ids',help='Optional comma-separated subset of the selected partition')
    parser.add_argument('--resume-collection',action='store_true',help='Skip completed matching episode files and retry incomplete ones')
    parser.add_argument('--eval-stage',choices=('before','after'),default='before')
    parser.add_argument('--checkpoint')
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args(argv)
    cfg,opsd=read_config(args.config)
    overrides={}
    for item in args.set:
        key,value=item.split('=',1);overrides[key]=json.loads(value)
    if overrides: cfg=load_settings(str(ROOT/'configs/rl-retail.json'),asdict(cfg)|overrides)
    opsd=validate_opsd_config(cfg,opsd)
    selected_task_ids=[item.strip() for item in args.task_ids.split(',') if item.strip()] if args.task_ids else None
    selection_partition='validation' if args.command=='evaluate' else args.partition
    if selected_task_ids and not set(selected_task_ids).issubset(training_task_ids(cfg,opsd,selection_partition)):
        parser.error('--task-ids must be members of the command partition')
    run=Path(args.run_dir).resolve() if args.run_dir else path(opsd['output_dir'])/('run-'+hashlib.sha256(os.urandom(16)).hexdigest()[:10])
    if args.command=='check':
        from openai import OpenAI
        ids=selected_task_ids or training_task_ids(cfg,opsd,args.partition)
        with OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=15,max_retries=0) as client:
            available=client.models.list().model_dump();models={m['id'] for m in available['data']}
        if cfg.user_model not in models: raise RuntimeError('Configured customer/hinter endpoint is unavailable')
        if not (path(cfg.initial_adapter)/'adapter_config.json').is_file(): raise RuntimeError('Initial retail SFT adapter is missing')
        print(json.dumps({'ready':True,'domain':'retail','partition':args.partition,'task_count':len(ids),
            'task_ids':ids,'rollouts_per_task':opsd['collection_rollouts_per_task'],
            'expected_collection_episodes':len(ids)*opsd['collection_rollouts_per_task'],
            'initial_adapter':str(path(cfg.initial_adapter)),'endpoint_model':cfg.user_model,
            'mode':opsd['mode'],
            'teacher_hints':('Per-turn Qwen diagnoses; teacher re-scores original sampled actions' if opsd['mode']=='hint' else 'Qwen hindsight diagnosis with replay from the original error state'),
            'max_hint_attempts_per_instance':opsd['max_hints_per_instance'],
            'retry_policy':None if opsd['mode']=='hint' else 'Generate a revised hint after each completed sub-1 reward recovery; replay the same state',
            'training_loss':'hint-conditioned frozen-policy top-100-plus-tail JSD + successful-state preservation',
            'train_partition_only':True,'wandb_mode':cfg.wandb_mode,'training_started':False},indent=2));return
    if not run.exists(): run.mkdir(parents=True)
    if args.command=='collect':
        atomic_json(run/'run-config.json',{'tau2':asdict(cfg),'opsd':opsd,'partition':args.partition})
    print('OPSD artifacts: '+str(run),flush=True)
    if args.command=='collect':
        from tau2_grpo.rl_policy import LocalPolicy
        policy=LocalPolicy(cfg,adapter=True,trainable=False)
        collect(policy,cfg,run,opsd,args.partition,selected_task_ids,args.resume_collection);return
    if args.command=='prepare-hint':
        from tau2_grpo.retail_hint import prepare
        rows=prepare(cfg,opsd,run)
        print(json.dumps({'prepared_hint_turns':len(rows)}));return
    if args.command=='diagnose':
        records=generate_hints(run,cfg,opsd)
        print(json.dumps({'diagnosed_candidates':len(records),'run_dir':str(run)}));return
    if args.command=='validate':
        from tau2_grpo.rl_policy import LocalPolicy
        candidates=json.loads((run/'candidates.json').read_text())
        policy=LocalPolicy(cfg,adapter=True,trainable=False)
        validate_candidates(policy,cfg,run,candidates,opsd);return
    if args.command=='train':
        import fcntl
        gpu_lock=open(ROOT/'outputs/.retail-training.lock','a')
        print('Waiting for retail training GPU lock',flush=True)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        # Training only uses the reserved RL partition. Validation tasks are
        # never accepted by the collection command used for training runs.
        manifest=json.loads(path(cfg.split_manifest).read_text())
        if manifest['partitions']['rl']!=task_ids(cfg,'rl'): raise RuntimeError('Tau2 RL partition changed')
        from tau2_grpo.rl_policy import LocalPolicy
        policy=LocalPolicy(cfg,adapter=True,trainable=True)
        train(policy,cfg,run,opsd)
    if args.command=='evaluate':
        evaluate(cfg,run,opsd,args.eval_stage,args.checkpoint,selected_task_ids)


if __name__=='__main__': main()
