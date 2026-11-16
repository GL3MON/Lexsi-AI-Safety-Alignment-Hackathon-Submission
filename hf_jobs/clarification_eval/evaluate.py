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
    def add(name,history,expected=None,clarify=None):
        rows.append(dict(name=name,messages=[dict(role='system',content=system)]+history,expected_calls=[expected] if expected else [],clarification=clarify))
    for n,(first,last,zip_code,email,oid) in enumerate([
        ('Elara','Mercer','63118','elara.mercer47@fastmail.com','#W8361942'),
        ('Ivo','Kovacs','97214','ivo.kovacs+receipts@proton.me','#W5927368')]):
        uid=f'{first.lower()}_{last.lower()}_5931'
        request=f'Please retrieve the current shipping status of order {oid}. Do not cancel or modify anything.'
        def user(text):return dict(role='user',content=text)
        add(f'{n}_missing_identity',[user(request)],clarify='identity')
        add(f'{n}_missing_zip',[user(f'{request} I cannot remember my email. My first name is {first} and last name is {last}. Use name and ZIP to find my account, but I have not provided my ZIP yet.')],clarify='zip')
        add(f'{n}_missing_last_name',[user(f'{request} I cannot remember my email. My first name is {first} and ZIP is {zip_code}. I have not provided my last name yet.')],clarify='last_name')
        add(f'{n}_missing_first_name',[user(f'{request} I cannot remember my email. My last name is {last} and ZIP is {zip_code}. I have not provided my first name yet.')],clarify='first_name')
        intro=[user(f'{request} My email is {email}.')]
        lookup=call('find_user_id_by_email',{'email':email})
        failed=intro+[lookup,tool('find_user_id_by_email','Error: User not found')]
        add(f'{n}_failed_lookup_missing_alternative',failed,clarify='identity')
        add(f'{n}_partial_answer_still_missing_zip',[
            user(f'{request} I cannot remember my email.'),
            dict(role='assistant',content='Please provide your first name, last name, and ZIP code so I can locate your account.'),
            user(f'My first name is {first} and my last name is {last}.')],clarify='zip')
        add(f'{n}_complete_email',intro,{'name':'find_user_id_by_email','arguments':{'email':email}})
        add(f'{n}_complete_name_zip',[user(f'{request} I cannot remember my email. My first name is {first}, last name {last}, and ZIP {zip_code}.')],{'name':'find_user_id_by_name_zip','arguments':{'first_name':first,'last_name':last,'zip':zip_code}})
        auth=intro+[lookup,tool('find_user_id_by_email',uid)]
        add(f'{n}_authenticated_get_profile',auth,{'name':'get_user_details','arguments':{'user_id':uid}})
        profile=auth+[call('get_user_details',{'user_id':uid}),tool('get_user_details',dict(user_id=uid,email=email,name=dict(first_name=first,last_name=last),orders=[oid]))]
        add(f'{n}_profile_retrieve_explicit_status',profile,{'name':'get_order_details','arguments':{'order_id':oid}})
        revised=f'{first.lower()}.{last.lower()}19@outlook.com'
        correction=failed+[dict(role='assistant',content='That email was not found. Please check your account email.'),user(f'The correct email is {revised}. Please use it to find my account and retrieve the order status.')]
        add(f'{n}_email_correction',correction,{'name':'find_user_id_by_email','arguments':{'email':revised}})
        recovery=profile+[call('get_order_details',{'orderID':oid}),tool('__runtime__',{'error':"Unexpected argument 'orderID'; required argument is 'order_id'. Please correct the field name."})]
        add(f'{n}_wrong_key_recovery',recovery,{'name':'get_order_details','arguments':{'order_id':oid}})
    assert len(rows)==24
    return rows


def grade(text,case,ended=True):
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
        note='Small fixed skill diagnostic with scripted tool histories; clarification checks are heuristic. Not a live Tau2 task-success benchmark or evidence of regression without a paired baseline.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
