"""Turn-level hindsight preparation on original sampled retail actions."""
import copy
import json
from pathlib import Path
from tau2_grpo.rl_rollout import atomic_json

REFLECTION = '''Review the completed retail conversation and terminal reward as evidence. Return JSON {"turns":[{"turn":0,"diagnosis":"brief causal assessment","correction":"optional safe advice"}]}. Use exactly the supplied integer turn numbers, at most one entry per turn. Diagnose each original decision using the policy and evidence: invented arguments, wrong tools, missed clarification, repeated reads, incomplete multi-order requests, and unsupported success claims. Correct decisions may receive a positive assessment. Omit turns with no useful signal. Do not invent identifiers or tools. Corrections are advice, not a new trajectory. The teacher will assess the ORIGINAL action from its ORIGINAL pre-action history.'''


def reflect_episode(client, tokenizer, cfg, episode, actions, cache=None):
    """Use the full trace when it fits, otherwise diagnose each local turn."""
    def request_for(payload):
        return [{'role':'system','content':REFLECTION},
                {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    def fits(request):
        return len(tokenizer.apply_chat_template(request,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False))+2048+256<=cfg.external_context_limit
    full=request_for({'conversation':episode['messages'],'terminal_reward':episode['reward'],
        'actions':[{'turn':a['turn'],'text':a['text']} for a in actions]})
    requests=[];errors=[]
    if fits(full):requests.append(('full',None,full))
    else:
        raw=episode['messages']
        for index,action in enumerate(actions):
            for radius in (2,1,0):
                left=max(0,index-radius);right=min(len(actions),index+radius+1)
                start=int(actions[left]['raw_message_count'])
                end=int(actions[right]['raw_message_count']) if right<len(actions) else len(raw)
                payload={'scope':'local neighborhood; omitted history is unknown, not evidence of missing facts',
                    'original_request':raw[1],'policy_and_tools':raw[0],
                    'terminal_reward':episode['reward'],'target_turn':action['turn'],
                    'conversation_neighborhood':raw[start:end],
                    'actions':[{'turn':action['turn'],'text':action['text']}],
                    'instruction':'Diagnose only target_turn. Use the nearby observations as evidence; do not infer that omitted actions never happened.'}
                request=request_for(payload)
                if fits(request):
                    requests.append(('neighborhood',action['turn'],request));break
            else:errors.append({'turn':action['turn'],'reason':'single_turn_neighborhood_context_exceeded'})
    annotations=[];audit=[]
    for scope,target,request in requests:
        try:
            key=json.dumps(request,sort_keys=True,ensure_ascii=False)
            if cache is not None and key in cache:
                from openai.types.chat import ChatCompletion
                response=ChatCompletion.model_validate(cache[key])
            else:
                response=client.chat.completions.create(model=cfg.user_model,messages=request,temperature=0,
                    max_tokens=2048,response_format={'type':'json_object'},
                    extra_body={'chat_template_kwargs':{'enable_thinking':False}})
                if cache is not None:cache[key]=response.model_dump()
            audit.append({'scope':scope,'target_turn':target,'request':request,'response':response.model_dump()})
            if response.choices[0].finish_reason=='length':raise ValueError('truncated reflection')
            result=json.loads(response.choices[0].message.content or '')['turns']
            if not isinstance(result,list):raise ValueError('turns must be a list')
            if target is not None and any(x.get('turn')!=target for x in result):raise ValueError('reflection returned non-target turn')
            annotations.extend(result)
        except Exception as error:
            errors.append({'turn':target,'reason':'reflection_failed: '+str(error)})
    return annotations,audit,errors


def windowed_teacher(controller, history, observations, reward, hint, completion_ids, limit):
    """Slide only teacher history; retain the exact sampled student context."""
    from tau2_grpo.hint_distillation import teacher_messages
    view=copy.deepcopy(history);removed=0;excerpted=False
    observation=observations
    while True:
        # Removing an exchange can join the opening and latest observation.
        # Merge adjacent user content instead of inventing an assistant turn.
        merged=[]
        for message in view:
            if merged and message['role']=='user' and merged[-1]['role']=='user':
                merged[-1]['content']+='\n\n[Later observation; intervening history omitted]\n'+message['content']
            else:merged.append(copy.deepcopy(message))
        view=merged
        result=controller.render(teacher_messages(view,observation,reward,hint))
        if len(result['prompt_ids'])+len(completion_ids)<=limit:
            return result,{'removed_history_messages':removed,'observation_excerpted':excerpted}
        # Keep system, the original opening, and the latest complete exchange.
        assistants=[i for i,m in enumerate(view) if m['role']=='assistant']
        if len(assistants)>1:
            left=assistants[0];right=assistants[1]
            removed+=right-left;del view[left:right]
            continue
        if len(assistants)==1 and len(view)>3:
            left=assistants[0]
            removed+=len(view)-left-1
            view=view[:left]+[view[-1]]
            continue
        if not excerpted:
            encoded=controller.tokenizer.encode(json.dumps(observations,ensure_ascii=False),add_special_tokens=False)
            observation={'scope':'excerpt of immediate observation; omitted content is unknown',
                'beginning':controller.tokenizer.decode(encoded[:256]),
                'ending':controller.tokenizer.decode(encoded[-256:])}
            excerpted=True;continue
        raise ValueError('teacher policy plus minimal turn feedback exceeds context')


def prepare(cfg, opsd, run):
    from openai import OpenAI
    from transformers import AutoTokenizer
    from tau2_grpo.hint_distillation import teacher_messages
    from tau2_grpo.rl_context import ContextController
    run=Path(run)
    tokenizer=AutoTokenizer.from_pretrained(cfg.model_id)
    external_tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
    controller=ContextController(tokenizer,None,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
    rows=[];audit=[];rejected=[];cache={}
    previous_files=list(run.glob('*/hint-reflection-audit.json'))+[run/'hint-reflection-audit.json']
    for previous in previous_files:
        if not previous.exists():continue
        for entry in json.loads(previous.read_text()):
            response=entry.get('response',{})
            if response.get('choices',[{}])[0].get('finish_reason')=='stop':
                cache[json.dumps(entry['request'],sort_keys=True,ensure_ascii=False)]=response
    with OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=180,max_retries=2) as client:
        for file in sorted((run/'collection').glob('episode-*.json')):
            episode=json.loads(file.read_text())
            if not episode.get('complete') or episode.get('reward') is None: continue
            actions=episode.get('actions',[])
            annotations,records,errors=reflect_episode(client,external_tokenizer,cfg,episode,actions,cache)
            audit.extend({'file':str(file),**record} for record in records)
            rejected.extend({'file':str(file),**error} for error in errors)
            atomic_json(run/'hint-reflection-audit.json',audit)
            try:
                valid={int(a['turn']):a for a in actions};seen=set();prepared=[]
                for ann in annotations:
                    t=ann.get('turn')
                    if type(t) is not int or t not in valid or t in seen:raise ValueError('unknown or duplicate turn')
                    seen.add(t)
                    diagnosis=ann.get('diagnosis');correction=ann.get('correction') or ''
                    if not isinstance(diagnosis,str) or not diagnosis.strip():continue
                    if not isinstance(correction,str):raise ValueError('invalid correction')
                    action=valid[t];index=actions.index(action)
                    if not action.get('ended_with_eos') or not action.get('completion_ids'):continue
                    history=copy.deepcopy(action['messages'])
                    if controller.render(history)['prompt_ids']!=action['prompt_ids']:
                        raise ValueError('saved student prompt does not match tokenizer')
                    start=int(action['raw_message_count'])+1
                    end=int(actions[index+1]['raw_message_count']) if index+1<len(actions) else len(episode['messages'])
                    observations=episode['messages'][start:end]
                    hint=diagnosis+ ('\nSuggested correction: '+correction if correction else '')
                    try:
                        teacher,window=windowed_teacher(controller,history,observations,episode['reward'],hint,
                            action['completion_ids'],cfg.context_limit)
                    except ValueError as error:
                        rejected.append({'file':str(file),'turn':t,'reason':str(error)});continue
                    prepared.append({'candidate_id':f"{episode['episode_id']}:turn-{t}",'episode_id':episode['episode_id'],
                        'task_id':str(episode['task_id']),'seed':episode['seed'],'turn':t,'reward':episode['reward'],
                        'diagnosis':diagnosis,'hint':hint,'teacher_window':window,'student_prompt_ids':action['prompt_ids'],
                        'teacher_prompt_ids':teacher['prompt_ids'],'student_completion_ids':action['completion_ids'],
                        'student_messages':history,'teacher_messages':teacher['messages'],'original_action':action['text']})
                for row in prepared:row['episode_turn_weight']=1/len(prepared)
                rows.extend(prepared)
            except (ValueError,KeyError,TypeError) as error:
                rejected.append({'file':str(file),'reason':str(error)})
            print(json.dumps({'event':'reflected','file':str(file),'prepared_turns':len(rows),'rejected':len(rejected)}),flush=True)
            (run/'hint_turns.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
            atomic_json(run/'hint-preparation-report.json',{'turns':len(rows),'episodes':len({x['episode_id'] for x in rows}),'rejected':rejected})
    return rows
