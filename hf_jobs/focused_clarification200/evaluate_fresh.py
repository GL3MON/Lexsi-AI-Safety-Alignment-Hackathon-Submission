"""Fixed, fresh tool-decision tests. Scripted histories; not a Tau2 task benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def call(name,arguments):
    return {'role':'assistant','content':'<tool_call>'+json.dumps(dict(name=name,arguments=arguments),separators=(',',':'))+'</tool_call>'}


def tool(name,value):
    return dict(role='tool',name=name,content=json.dumps(dict(name=name,content=value),separators=(',',':')))


def cases(system):
    rows=[]
    def add(name,history,expected=None,kind=None):
        rows.append(dict(name=name,messages=[dict(role='system',content=system)]+history,expected_calls=[expected] if expected else [],clarification=kind))
    people=[('Safiya','Petrenko','04852','safiya.petrenko64@mail.example.net','#W6184732'),('Luca','Benali','29416','luca_benali+shop@example.org','#W3758196'),('Thora','Sundberg','83726','thora-sundberg91@example.com','#W9472531')]
    for n,(first,last,zip_code,email,oid) in enumerate(people):
        uid=f'customer_{first.lower()}_8642'
        request=f"I'd like the current status of order {oid}. Please leave the order unchanged."
        def user(text):return dict(role='user',content=text)
        add(f'{n}_missing_identity',[user(request)],kind='identity')
        add(f'{n}_missing_zip',[user(request+f' I cannot recall my email. My given name is {first} and my surname is {last}.')],kind='zip')
        add(f'{n}_missing_last_name',[user(request+f' I cannot recall my email. My first name is {first}; my ZIP is {zip_code}.')],kind='last_name')
        add(f'{n}_missing_first_name',[user(request+f' I cannot recall my email. My surname is {last}; my ZIP is {zip_code}.')],kind='first_name')
        intro=[user(request+f' Start by finding my account with this email: {email}.')]
        lookup=call('find_user_id_by_email',{'email':email})
        failed=intro+[lookup,tool('find_user_id_by_email','Error: User not found')]
        add(f'{n}_failed_lookup_missing_alternative',failed,kind='identity')
        add(f'{n}_partial_answer_still_missing_zip',[user(request+' I cannot remember my email.'),dict(role='assistant',content='What are your first name, last name, and ZIP code?'),user(f'{first} {last} is my full name.')],kind='zip')
        add(f'{n}_complete_email',intro,dict(name='find_user_id_by_email',arguments={'email':email}))
        add(f'{n}_complete_name_zip',[user(request+f' Please find my account by name and ZIP: first name {first}, last name {last}, ZIP {zip_code}. I cannot remember my email.')],dict(name='find_user_id_by_name_zip',arguments=dict(first_name=first,last_name=last,zip=zip_code)))
        auth=intro+[lookup,tool('find_user_id_by_email',uid)]
        add(f'{n}_authenticated_get_profile',auth,dict(name='get_user_details',arguments={'user_id':uid}))
        profile=auth+[call('get_user_details',{'user_id':uid}),tool('get_user_details',dict(user_id=uid,email=email,name=dict(first_name=first,last_name=last),orders=[oid]))]
        add(f'{n}_profile_retrieve_explicit_status',profile,dict(name='get_order_details',arguments={'order_id':oid}))
        corrected=f'{first.lower()}.{last.lower()}72@example.org'
        add(f'{n}_email_correction',failed+[dict(role='assistant',content='That email did not match an account. What is the correct account email?'),user(f'Use {corrected} instead. Please look me up with that corrected email, then check the order status.')],dict(name='find_user_id_by_email',arguments={'email':corrected}))
        add(f'{n}_wrong_key_recovery',profile+[call('get_order_details',{'orderID':oid}),tool('__runtime__',{'error':"Required argument 'order_id' missing; argument 'orderID' is unsupported. Retry using the required field name."})],dict(name='get_order_details',arguments={'order_id':oid}))
    assert len(rows)==36
    return rows


def original_grade(text,case,ended=True):
    blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S);parsed=[];errors=[]
    if text.count('<tool_call>')!=len(blocks) or text.count('</tool_call>')!=len(blocks):errors.append('unbalanced_tags')
    for block in blocks:
        try:
            value=json.loads(block)
            if not isinstance(value,dict) or set(value)!={'name','arguments'} or not isinstance(value['arguments'],dict):raise ValueError('invalid_shape')
            parsed.append(value)
        except ValueError:errors.append('invalid_call_json_or_shape')
    expected=case['expected_calls'];passed=not errors and ended
    if expected:passed=passed and parsed==expected
    else:
        passed=passed and not parsed
        lower=text.lower();kind=case['clarification']
        if kind=='identity':passed=passed and bool(re.search(r'email|\bzip\b|postal|first name|last name',lower)) and ('?' in text or 'please' in lower)
        elif kind=='zip':passed=passed and bool(re.search(r'\bzip\b|postal',lower))
        elif kind=='first_name':passed=passed and bool(re.search(r'first name|given name|full name',lower))
        elif kind=='last_name':passed=passed and bool(re.search(r'last name|surname|family name|full name',lower))
        elif kind=='confirmation':passed=passed and bool(re.search(r'confirm|proceed|would you like',lower)) and not bool(re.search(r'i have cancel|successfully cancel|has been cancel',lower))
        elif kind=='final':passed=passed and bool(re.search(r'cancel',lower)) and '83.75' in text
    return dict(passed=bool(passed),parsed_calls=parsed,format_errors=errors)


def grade(text,case,ended=True):
    result=original_grade(text,case,ended)
    if case['expected_calls']:return result
    lower=text.lower();kind=case['clarification'];issues=[]
    requests=re.findall(r'(?:could you|can you|would you|please|what (?:is|are)|which|may i)[^?.!]*(?:[?.!]|$)',lower)
    request=' '.join(requests)
    fields={'first_name':r'first name|given name','last_name':r'last name|surname|family name','zip':r'\bzip\b|postal','email':r'e-?mail'}
    if kind in fields:
        if not re.search(fields[kind],request):issues.append('missing_specific_field_request')
        for other,pattern in fields.items():
            if other!=kind and re.search(pattern,request):issues.append('asks_known_or_unavailable_field_'+other)
        if re.search(r'other|additional|relevant|anything else|full name',request):issues.append('generic_extra_request')
    if re.search(r'i (?:have |already )?(?:found|located|verified|authenticated)|here are (?:the|your) (?:order|account)|shipping address|gift.card|balance|order status:|status is (?:pending|delivered|cancelled)',lower):issues.append('unsupported_success_or_result_claim')
    if not requests:issues.append('no_clear_question_or_request')
    result['clarification_issues']=issues;result['passed']=result['passed'] and not issues
    return result

def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from peft import PeftModel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--system',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_step=json.loads((args.checkpoint/'trainer_state.json').read_text())['global_step']
    if not isinstance(checkpoint_step,int) or checkpoint_step<1:raise ValueError('Invalid source checkpoint step')
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
    assert tokenizer.chat_template==(args.checkpoint/'chat_template.jinja').read_text()
    model=AutoModelForCausalLM.from_pretrained('GL3MON/tiny-aya-global-function-calling-sft',
        revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(model,args.checkpoint,is_trainable=False);model.eval()
    data=cases(args.system.read_text());results=[]
    settings=GenerationConfig(do_sample=False,max_new_tokens=256,eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,use_cache=True)
    for index,case in enumerate(data):
        inputs=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,
            return_tensors='pt',return_dict=True).to('cuda:0')
        if inputs['input_ids'].shape[1]+256>8192:raise RuntimeError('Evaluation prompt exceeds 8K reserve')
        started=time.monotonic()
        with torch.inference_mode():tokens=model.generate(**inputs,generation_config=settings,logits_to_keep=1)[0,inputs['input_ids'].shape[1]:]
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        result=dict(name=case['name'],messages=case['messages'],expected_calls=case['expected_calls'],
            clarification=case['clarification'],text=text,completion_ids=tokens.tolist(),
            prompt_tokens=inputs['input_ids'].shape[1],seconds=time.monotonic()-started,
            **grade(text,case,bool(len(tokens) and tokens[-1].item()==tokenizer.eos_token_id)))
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        (args.output/'conversations.txt').write_text('\n\n'.join(
            f"=== {r['name']} | PASS={r['passed']} ===\n"+'\n\n'.join(
                m['role'].upper()+':\n'+m['content'] for m in r['messages'] if m['role']!='system')+
                '\n\nGENERATED ASSISTANT:\n'+r['text']+'\nEXPECTED CALLS: '+json.dumps(r['expected_calls']) for r in results))
        print(json.dumps(dict(case=index+1,name=result['name'],passed=result['passed'],text=text)),flush=True)
    report=dict(checkpoint=checkpoint_step,total=len(results),passed=sum(r['passed'] for r in results),
        exact_tool_decisions_passed=sum(r['passed'] for r in results if r['expected_calls']),
        exact_tool_decisions=sum(bool(r['expected_calls']) for r in results),
        temperature=0,backend='transformers',dtype='float16',gpu=torch.cuda.get_device_name(0),
        adapter_sha256=hashlib.sha256((args.checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        note='Fresh fixed skill diagnostic; request-field and unsupported-claim checks are heuristic and require output review. Not a live Tau2 task-success benchmark or evidence of regression without a paired baseline.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
