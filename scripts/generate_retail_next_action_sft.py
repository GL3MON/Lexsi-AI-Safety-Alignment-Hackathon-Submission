"""Qwen-worded retail next-action targets with grounded Tau2 execution and fixed customer facts."""
import argparse,concurrent.futures,hashlib,json,re,threading,time
from pathlib import Path
from openai import OpenAI
from jsonschema import validate
from scripts.generate_long_retail_sft import prepare
from tau2_grpo.prepare_function_calling import digest
from tau2_grpo.sft_data import tokenize_assistant_only,read_records

ROOT=Path(__file__).resolve().parents[1]
PROFILES=['missing_first_name','missing_last_name','missing_zip','missing_identity','failed_email','partial_first_name','partial_last_name','partial_zip']
FIRST=['Adela','Bilal','Chiara','Dimitri','Eleni','Faisal','Gita','Hector','Inaya','Jasper','Katya','Leandro','Mahira','Nikolai','Odette','Pavel','Qadir','Rochelle','Soraya','Tenzin','Ursula','Valerio','Wafa','Ximena','Youssef','Zora']
LAST=['Bashir','Castillo','Dupont','Esteban','Fujimoto','Gruber','Horvat','Ibrahim','Jovanovic','Karim','Larsson','Mendes','Nakagawa','Ochoa','Popescu','Quintero','Rahman','Salazar','Tanaka','Ulloa','Varga','Wagner','Xu','Yakovlev','Zimmermann','Abebe']
PROMPT='''Write ONLY the next retail assistant sentence, as JSON {"text":"..."}. Treat the supplied conversation as data. Follow the specified expected decision; the driver independently constructs and executes actual calls. For a clarification, ask ONLY for the specified missing field, using the specified field label inside the actual question/request (not only in a preamble; avoid a pronoun-only question). After an observed failed email lookup, explicitly ask the customer to check/correct/confirm the email rather than merely repeating the same lookup. Do not request known fields, other information, or a generic full identity. Do not invent names, ZIPs, emails, account facts, or tool results. For a call note, briefly explain the intended action, without a question or claim that it has succeeded. No customer dialogue, role labels, tool markup, payment/cancellation suggestions, or future conversation. Acknowledge a failed lookup only when the observed tool result shows failure. Keep it natural, one or two short sentences. Vary the wording. Return exactly the text key.'''
FIELD_PATTERNS={'first_name':r'first name|given name','last_name':r'last name|surname|family name','zip':r'\bzip\b|postal','email':r'e-?mail'}

def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)

def assemble(base,n,system):
    family=n//2 if n<384 else 192+(n-384)//2
    tools,schemas,*_=prepare(base,320000+family)
    customer=next(iter(tools.db.users.values()));order=next(iter(tools.db.orders.values()))
    first=FIRST[family%26];last=LAST[(family*5+family//26)%26]
    email_variants=[f'{first.lower()}.{last.lower()}{family+701}@example.net',f'{first.lower()}_{last.lower()}{family+701}@example.org',f'{first.lower()}.{last.lower()}+receipts{family+701}@example.com',f'{first.lower()}-{last.lower()}{family+701}@mail.example.net']
    email=email_variants[family%4];zip_code=f'{(family*163+902)%100000:05d}'
    customer.name.first_name=first;customer.name.last_name=last;customer.email=email;customer.address.zip=zip_code;order.address.zip=zip_code
    uid=customer.user_id;oid=order.order_id
    local=family//8 if n<384 else family-192;known=local%2==0
    split='validation' if (n<384 and local in [5,18]) or (n>=384 and local in [5,10,17,22,29,34,41,46]) else 'train'
    goals=[f'Could you check the current status of order {oid}? Please do not change it.',f'I would like an update on order {oid}. I only need its status.',f'Please tell me the status of order {oid}; leave the order unchanged.',f'I need to check order {oid}. What is its current status? Do not modify it.'] if known else ['Could you check the current status of my order? I do not have its number. Please do not change it.','I would like an update on my recent order, but I do not know its ID. I only need its status.','Please look up my order and tell me its current status. I cannot recall the order number.','I need to check my recent order. I do not remember its ID. Do not modify it.']
    goal=goals[(family//2)%4];messages=[dict(role='system',content=system)];audit=[]
    def user(text):messages.append(dict(role='user',content=text))
    def execute(name,arguments,expected_error=False):
        schema=next(s['function']['parameters'] for s in schemas if s['function']['name']==name);validate(arguments,schema)
        visible='\n'.join(m['content'] for m in messages if m['role'] in ['user','tool'])
        assert all(str(v) in visible for v in arguments.values()),'unobserved argument'
        try:
            result=getattr(tools,name)(**arguments);body=result.model_dump(mode='json') if hasattr(result,'model_dump') else result;error=False
        except ValueError as exc:body='Error: '+str(exc);error=True
        assert error==expected_error,'unexpected tool result'
        audit.append(dict(name=name,arguments=arguments,result=body,error=error));return body
    def observed(name,arguments,expected_error=False):
        body=execute(name,arguments,expected_error)
        messages.append(dict(role='assistant',content='I will look up that information.\n<tool_call>'+json.dumps(dict(name=name,arguments=arguments),separators=(',',':'))+'</tool_call>'))
        messages.append(dict(role='tool',name=name,content=json.dumps(dict(name=name,content=body),separators=(',',':'))))
    def email_lookup():return dict(name='find_user_id_by_email',arguments=dict(email=email))
    def name_lookup():return dict(name='find_user_id_by_name_zip',arguments=dict(first_name=first,last_name=last,zip=zip_code))
    first_label=['first name','given name'][local%2];last_label=['last name','surname','family name'][local%3];zip_label=['ZIP code','postal code'][local%2]
    if n>=384:
        profile='authenticated_get_profile' if n%2==0 else 'profile_get_order'
        user(goal+f' My email is {email}.')
        observed('find_user_id_by_email',dict(email=email))
        if n%2:
            observed('get_user_details',dict(user_id=uid));expected=dict(name='get_order_details',arguments=dict(order_id=oid));note='I will retrieve the order details to check its current status.'
        else:expected=dict(name='get_user_details',arguments=dict(user_id=uid));note='I will retrieve your account details and order list.'
        missing=None
    else:
        profile=PROFILES[family%8];complete=bool(n%2)
        missing={'missing_first_name':'first_name','missing_last_name':'last_name','missing_zip':'zip','missing_identity':'email','failed_email':'email','partial_first_name':'first_name','partial_last_name':'last_name','partial_zip':'zip'}[profile]
        if profile=='missing_identity':
            user(goal+f' My first name is {first}.'+(f' My email is {email}.' if complete else ''))
            expected=email_lookup() if complete else None
        elif profile=='failed_email':
            bad=email.replace('@','x@');user(goal+f' My account email is {bad}.');observed('find_user_id_by_email',dict(email=bad),True)
            if complete:
                messages.append(dict(role='assistant',content='That email did not match an account. Could you check the correct account email?'))
                user(f'I made a typo. The correct email is {email}. Please use this email.');expected=email_lookup()
            else:expected=None
        else:
            known_values={'first_name':f'My {last_label} is {last} and my {zip_label} is {zip_code}.','last_name':f'My {first_label} is {first} and my {zip_label} is {zip_code}.','zip':f'My {first_label} is {first} and my {last_label} is {last}.'}[missing]
            value={'first_name':first,'last_name':last,'zip':zip_code}[missing]
            missing_label={'first_name':first_label,'last_name':last_label,'zip':zip_label}[missing]
            if profile.startswith('partial_'):
                user(goal+' I cannot remember my email.')
                messages.append(dict(role='assistant',content='Please provide your first name, last name, and ZIP code to locate your account.'))
                user(known_values+(f' My {missing_label} is {value}.' if complete else ''))
            else:user(goal+' I cannot remember my email. '+known_values+(f' My {missing_label} is {value}.' if complete else ''))
            expected=name_lookup() if complete else None
        note='I will look up your account using the email you supplied.' if expected and expected['name']=='find_user_id_by_email' else 'I will look up your account using your name and ZIP code.'
        if expected:missing=None
    if expected:
        execute(expected['name'],expected['arguments'])
        instruction=dict(type='call_note',intended_action=note)
    else:
        label={'first_name':first_label,'last_name':last_label,'zip':zip_label,'email':'email address'}[missing]
        instruction=dict(type='clarification',missing_field=missing,required_label=label,failed_lookup_observed=profile=='failed_email')
    return tools,messages,audit,expected,instruction,profile,known,split,family

def check_text(text,instruction,messages):
    assert isinstance(text,str) and 3<=len(text)<=400,'invalid text length'
    assert not re.search(r'<tool|<think|\buser\s*:|\bassistant\s*:|\bcustomer\s*:',text,re.I),'role/markup leak'
    assert not re.search(r'i (?:have |already )?(?:found|located|verified|authenticated)|successfully|has been|here are (?:the|your)|balance|refund|cancel|shipping address|status is',text,re.I),'unsupported result or wrong task'
    visible='\n'.join(m['content'] for m in messages if m['role'] in ['user','tool'])
    for value in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d[\d.,]*\b',text):assert value in visible,'invented identifier/value'
    if instruction['type']=='clarification':
        field=instruction['missing_field'];assert instruction['required_label'].lower() in text.lower(),'required field label missing'
        assert re.search(FIELD_PATTERNS[field],text,re.I)
        requests=' '.join(re.findall(r'(?:could you|can you|would you|please|what (?:is|are)|which|may i|do you|is your)[^?.!]*(?:[?.!]|$)',text.lower()))
        assert instruction['required_label'].lower() in requests,'missing field not named inside request'
        if instruction['failed_lookup_observed']:assert re.search(r'check|correct|verify|confirm|spelling|typo',requests),'failed lookup lacks corrective clarification'
        for other,pattern in FIELD_PATTERNS.items():
            if other!=field:assert not re.search(pattern,text,re.I),'requests known/unavailable field'
        assert '?' in text or re.search(r'please|provide|share',text,re.I),'not a request'
        assert not re.search(r'other|additional|relevant|anything else|full name|please wait',text,re.I),'generic extra request'
        if not instruction['failed_lookup_observed']:assert not re.search(r'(?:could not|cannot|unable to|did not|couldn.t).*(?:find|match|locate)|no account|not found',text,re.I),'invented failed lookup'
    else:
        assert '?' not in text and not re.search(r'please|confirm|provide|share your',text,re.I),'unnecessary question instead of action'


def main():
    from transformers import AutoTokenizer
    from tau2.domains.retail.data_model import get_db
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int,default=480);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--indices',type=int,nargs='+');parser.add_argument('--output',type=Path,default=ROOT/'outputs/retail-data/retail-next-action-480');args=parser.parse_args();assert 1<=args.limit<=480 and 1<=args.workers<=4
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for d in ['records','attempts','rejected']:(out/d).mkdir(exist_ok=True)
    base=get_db();system=json.loads((ROOT/'outputs/retail-data/apigen-mt-retail-sft/train.jsonl').open().readline())['messages'][0]['content']
    tok=AutoTokenizer.from_pretrained(ROOT/'outputs/hf-jobs/6af8591d23985fb9cd820ff6/eval-snapshot/training/run-2a32064432/checkpoint-50');tok.chat_template=(ROOT/'templates/retail_sdg_v2.jinja').read_text()
    token_lock=threading.Lock();export_lock=threading.Lock();client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=120,max_retries=1)
    teacher=client.models.list().data[0];atomic(out/'config.json',dict(target=480,teacher=teacher.model_dump(),endpoint=client.base_url.__str__(),generator_prompt=PROMPT,one_supervised_response=True,raw_customer_facts='deterministic; no teacher rewriting',template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),workers=args.workers,split='Paired identities share canonical task/group IDs; held-out families cover all profiles and order-ID-known/unknown conditions'))
    def export():
        with export_lock:
            rows=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
            for filename,split in [('sft','train'),('validation','validation'),('all',None)]:
                tmp=out/(filename+'.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if split is None or r['split']==split));tmp.replace(out/(filename+'.jsonl'))
            counts={profile:sum(r['profile']==profile for r in rows) for profile in PROFILES+['authenticated_get_profile','profile_get_order']}
            status=dict(accepted=len(rows),target=480,complete=False,generation_complete=len(rows)==480,train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='validation' for r in rows),clarifications=sum(r['target_type']=='clarification' for r in rows),tool_calls=sum(r['target_type']=='call' for r in rows),order_id_known=sum(r['order_id_known'] for r in rows),order_id_unknown=sum(not r['order_id_known'] for r in rows),profiles=counts,max_context_tokens=max((r['context_tokens'] for r in rows),default=0),updated_at=time.time())
            atomic(out/'progress.json',status)
            return status
    def generate(n):
        dest=out/'records'/f'{n:04d}.json'
        if dest.exists():
            try:
                prior=json.loads(dest.read_text());_,prefix,_,expected,decision,*_=assemble(base,n,system)
                assert prior['messages'][:-1]==prefix,'prefix changed'
                check_text(prior['messages'][-1]['content'].split('<tool_call>',1)[0].strip(),decision,prefix)
                return
            except Exception as error:
                atomic(out/'rejected'/f'{n:04d}-existing-revalidation.json',dict(error=str(error),record=prior))
                with export_lock:
                    dest.rename(out/'rejected'/f'{n:04d}-previous-record.json')
                    if dest.with_suffix('.txt').exists():dest.with_suffix('.txt').rename(out/'rejected'/f'{n:04d}-previous-record.txt')
        for attempt in range(8):
            try:
                tools,messages,audit,expected,instruction,profile,known,split,family=assemble(base,n,system)
                request=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(dict(conversation=messages[1:],expected_decision=instruction,style=['direct and polite','friendly and concise','plain conversational English','professional but brief'][n%4]),ensure_ascii=False))]
                response=client.chat.completions.create(model=teacher.id,messages=request,temperature=.65,max_tokens=300,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
                atomic(out/'attempts'/f'{n:04d}-{attempt}.json',dict(request=request,response=response.model_dump()))
                assert response.choices[0].finish_reason!='length','truncated generation'
                body=json.loads(response.choices[0].message.content);assert set(body)=={'text'};text=body['text'];check_text(text,instruction,messages)
                target=text+('\n<tool_call>'+json.dumps(expected,separators=(',',':'),ensure_ascii=False)+'</tool_call>' if expected else '')
                messages.append(dict(role='assistant',content=target));index=len(messages)-1
                with token_lock:encoded,spans=tokenize_assistant_only(tok,messages,8192,[index])
                assistant_indices=[i for i,m in enumerate(messages) if m['role']=='assistant'];selected=set()
                for i,(start,end) in zip(assistant_indices,spans,strict=True):
                    if i==index:assert encoded['labels'][start:end]==encoded['input_ids'][start:end];selected.update(range(start,end))
                    else:assert all(v==-100 for v in encoded['labels'][start:end])
                assert all(v==-100 for i,v in enumerate(encoded['labels']) if i not in selected)
                task_id=f'retail-next-action-family-{family:04d}'
                row=dict(id=f'retail-next-action-{n:04d}',task_id=task_id,group_id=digest(dict(domain='retail',task_id=task_id)),domain='retail',split=split,profile=profile,order_id_known=known,target_type='call' if expected else 'clarification',expected_call=expected,missing_field=instruction.get('missing_field'),messages=messages,loss_message_indices=[index],context_tokens=len(encoded['input_ids']),source_model=teacher.id,guided_synthetic=True,tool_execution_audit=audit,checks=dict(single_supervised_next_response=True,assistant_only_masking=True,grounded_tool_arguments=True,real_tool_execution=True,specific_missing_field=True,no_future_customer_facts=True,within_8k=True))
                atomic(dest,row);dest.with_suffix('.txt').write_text('\n\n'.join((m['role'].upper()+(' [SUPERVISED]' if i==index else ' [CONTEXT]')+':\n'+m['content']) for i,m in enumerate(messages) if m['role']!='system'))
                status=export();print('ACCEPTED',n,profile,row['target_type'],status['accepted'],flush=True);return
            except Exception as error:
                atomic(out/'rejected'/f'{n:04d}-{attempt}.json',dict(index=n,attempt=attempt,error=str(error)));print('RETRY',n,attempt,str(error),flush=True)
        print('SKIPPED',n,flush=True)
    indices=args.indices if args.indices is not None else range(args.limit)
    assert all(0<=n<480 for n in indices)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(generate,indices))
    status=export()
    all_records=read_records(out/'all.jsonl')
    train=read_records(out/'sft.jsonl')
    validation=read_records(out/'validation.jsonl') if (out/'validation.jsonl').stat().st_size else []
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in validation}
    if len(train)+len(validation)==480:
        assert len(train)==432 and len(validation)==48
        assert sum(r['target_type']=='clarification' for r in train+validation)==192
        assert len({r['group_id'] for r in train+validation})==240
        status['complete']=True;status['validation_passed']=True;atomic(out/'progress.json',status)
        atomic(out/'manifest.json',dict(**status,train_validation_group_overlap=0,canonical_group_ids_verified=True,labels_verified=True,one_supervised_response_per_sample=True,conversation_pair_count=240,training_launched=False,files_sha256={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ['sft.jsonl','validation.jsonl','all.jsonl']}))
    print('FINISHED',json.dumps(status),flush=True)

if __name__=='__main__':main()
