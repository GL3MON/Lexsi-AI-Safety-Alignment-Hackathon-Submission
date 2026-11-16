"""Convert APIGen retail to verified Aya labels; retain observable recoveries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_TOKENIZER=None


def initialize_worker():
    from transformers import AutoTokenizer
    global _TOKENIZER
    _TOKENIZER=AutoTokenizer.from_pretrained('outputs/rl-glaive/run-55f242da3a/checkpoint-17/policy')
    _TOKENIZER.chat_template=Path('templates/retail_sdg_v2.jinja').read_text()


def convert_worker(item):
    index,line=item;row=json.loads(line)
    try:record,issue,encoded=normalize(row,index,_TOKENIZER,8192)
    except (ValueError,KeyError,TypeError) as error:
        record=None;encoded=None;issue=dict(index=index,reasons=[dict(reason='conversion',detail=str(error))])
    return index,row,record,issue,encoded


def error_response(value):
    if isinstance(value,dict) and value.get('error'):return True
    return isinstance(value,str) and bool(re.search(r'\berror\b|not found|does not exist|unknown (?:tool|function)|invalid (?:argument|parameter)',value,re.I))


def normalize(row,index,tokenizer,max_length):
    from jsonschema import Draft7Validator
    from tau2_grpo.prepare_function_calling import digest,prompt_group
    tools=json.loads(row['tools']) if isinstance(row['tools'],str) else row['tools']
    catalog={t['name']:t for t in tools if t['name']!='think'}
    validators={name:Draft7Validator(tool['parameters']) for name,tool in catalog.items()}
    calls={};issues=[];decoded=0
    source=row['conversations']
    for position,message in enumerate(source):
        if message['from']!='function_call':continue
        call=json.loads(message['value'])
        if isinstance(call['arguments'],str):call['arguments']=json.loads(call['arguments']);decoded+=1
        calls[position]=call
    invalid={}
    for position,call in calls.items():
        if call['name']=='think':continue
        if call['name'] not in validators:invalid[position]='unlisted_tool'
        elif not validators[call['name']].is_valid(call['arguments']):invalid[position]='argument_schema'
    for position,kind in invalid.items():
        following=source[position+1] if position+1<len(source) else None
        feedback=following['value'] if following and following['from']=='observation' else ''
        try:feedback=json.loads(feedback)
        except (ValueError,TypeError):pass
        next_calls=[p for p,c in calls.items() if p>position and c['name']!='think']
        repair=next((p for p in next_calls if p not in invalid),None)
        repair_feedback=source[repair+1]['value'] if repair is not None and repair+1<len(source) and source[repair+1]['from']=='observation' else None
        try:repair_feedback=json.loads(repair_feedback)
        except (ValueError,TypeError):pass
        if not error_response(feedback) or repair is None or repair_feedback is None or error_response(repair_feedback):
            issues.append(dict(source_turn=position,reason=kind,call=call,
                               detail='No observable error followed by a schema-valid successful correction'))
    if issues:return None,dict(index=index,reasons=issues),None
    system=row['system']+'\n\n<tools>'+json.dumps([{'type':'function','function':t} for t in catalog.values()],ensure_ascii=False,separators=(',',':'))+'</tools>'
    messages=[dict(role='system',content=system)];targets=[];pending=None;skip_observation=False;recoveries=[]
    for position,message in enumerate(source):
        role=message['from'];content=message['value']
        if role=='function_call':
            call=calls[position]
            if call['name']=='think':skip_observation=True;continue
            pending=call['name']
            messages.append(dict(role='assistant',content='<tool_call>'+json.dumps(call,ensure_ascii=False,separators=(',',':'))+'</tool_call>'))
            if position not in invalid:targets.append(len(messages)-1)
            else:recoveries.append(dict(message_index=len(messages)-1,source_turn=position,reason=invalid[position]))
        elif role=='observation':
            if skip_observation:skip_observation=False;continue
            if pending is None:raise ValueError('orphan_tool_response')
            try:value=json.loads(content)
            except ValueError:value=content
            messages.append(dict(role='tool',name=pending,content=json.dumps(dict(name=pending,content=value),ensure_ascii=False,separators=(',',':'))));pending=None
        elif role=='human':messages.append(dict(role='user',content=content))
        elif role=='gpt':
            if messages[-1]['role']=='assistant' and len(messages)-1 in targets:
                messages[-1]['content']+='\n'+content
            else:
                messages.append(dict(role='assistant',content=content));targets.append(len(messages)-1)
        else:raise ValueError('unknown_source_role:'+role)
    count=len(tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False))
    if count>max_length:return None,dict(index=index,reasons=[dict(reason='overlength',tokens=count,limit=max_length)]),None
    from tau2_grpo.sft_data import tokenize_assistant_only
    encoded,spans=tokenize_assistant_only(tokenizer,messages,max_length,targets)
    expected=sum(len(s['content']) for s in messages if s['role']=='assistant')
    if not expected:raise ValueError('empty_targets')
    record=dict(id=f'apigen-retail-{index:05d}',domain='retail',group_id=prompt_group(messages),
        messages=messages,loss_message_indices=targets,context_tokens=count,
        chat_template_path='templates/retail_sdg_v2.jinja',source_index=index,
        masked_faults=recoveries,decoded_argument_strings=decoded)
    return record,None,encoded


def main():
    from tau2_grpo.sft_data import split_validation,read_records
    from tau2_grpo.prepare_function_calling import digest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/retail-data/apigen-mt-retail/retail.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('outputs/retail-data/apigen-mt-retail-sft'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    accepted=[];rejected=[];seen=set();worst=None;stats=Counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool, (args.output/'quarantine.jsonl').open('w') as quarantine:
        for index,row,record,issue,encoded in pool.map(convert_worker,enumerate(args.source.open()),chunksize=4):
            if record:
                key=digest(record['messages'])
                if key in seen:record=None;issue=dict(index=index,reasons=[dict(reason='duplicate')])
                else:seen.add(key)
            if record:
                accepted.append(record);stats['retained_recovery_conversations']+=bool(record['masked_faults'])
                stats['masked_failed_calls']+=len(record['masked_faults']);stats['decoded_argument_strings']+=record['decoded_argument_strings']
                if worst is None or len(encoded['input_ids'])>len(worst['input_ids']):worst=encoded
            else:
                rejected.append(issue);quarantine.write(json.dumps(dict(**issue,source_record=row),ensure_ascii=False)+'\n')
            if (index+1)%100==0:print(json.dumps(dict(processed=index+1,accepted=len(accepted),quarantined=len(rejected))),flush=True)
    train,validation=split_validation(accepted,.05,42)
    for name,rows in [('train',train),('validation',validation)]:
        with (args.output/(name+'.jsonl')).open('w') as file:
            for row in rows:file.write(json.dumps(row,ensure_ascii=False)+'\n')
        read_records(args.output/(name+'.jsonl'))
    (args.output/'longest-tokenized.json').write_text(json.dumps(worst))
    reasons=Counter(item['reason'] for issue in rejected for item in issue['reasons'])
    report=dict(source=str(args.source),source_revision=json.loads((args.source.parent/'manifest.json').read_text())['source_revision'],
        total=index+1,accepted=len(accepted),train=len(train),validation=len(validation),quarantined=len(rejected),
        quarantine_reasons=dict(reasons),max_tokens=max(r['context_tokens'] for r in accepted),
        assistant_only_masks_verified=True,template_sha256=hashlib.sha256(Path('templates/retail_sdg_v2.jinja').read_bytes()).hexdigest(),
        train_validation_group_overlap=len({r['group_id'] for r in train}&{r['group_id'] for r in validation}),**stats)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
