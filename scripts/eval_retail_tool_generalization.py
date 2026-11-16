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
    email='neris.voss28@proton.me';new_email='neris.voss@outlook.com';uid='neris_voss_4827';oid='#W7419286';other='#W2846917'
    intro=[dict(role='user',content=f'My email is {email}. Please help me with order {oid}.')]
    auth=intro+[call('find_user_id_by_email',{'email':email}),tool('find_user_id_by_email',uid)]
    user=dict(user_id=uid,email=email,name=dict(first_name='Neris',last_name='Voss'),orders=[oid,other],
        payment_methods={'credit_card_6410287':dict(id='credit_card_6410287',source='credit_card',last_four='3084')})
    profile=auth+[call('get_user_details',{'user_id':uid}),tool('get_user_details',user)]
    item=dict(name='Bluetooth Speaker',item_id='5836291740',product_id='1749362850',price=83.75,options={'color':'blue'})
    order=dict(order_id=oid,user_id=uid,status='pending',items=[item],
        payment_history=[dict(transaction_type='payment',amount=83.75,payment_method_id='credit_card_6410287')])
    pending=profile+[call('get_order_details',{'order_id':oid}),tool('get_order_details',order)]
    rows=[]
    def add(name,history,expected=None,clarify=None):
        rows.append(dict(name=name,messages=[dict(role='system',content=system)]+history,
                         expected_calls=[expected] if expected else [],clarification=clarify))
    add('missing_identity',[dict(role='user',content='I need help locating an order.')],clarify='identity')
    add('missing_zip',[dict(role='user',content="I'm Neris Voss. I cannot remember my email. Please look up my account by name.")],clarify='zip')
    add('missing_last_name',[dict(role='user',content='My first name is Neris and my ZIP is 46217. I cannot remember my email. Use my name and ZIP.')],clarify='last_name')
    for name,address in [('email_copy_simple',email),('email_copy_plus','neris.voss+orders@proton.me'),
                         ('email_copy_underscore','n_voss28@outlook.com'),('email_copy_hyphen','neris-voss@fastmail.com')]:
        add(name,[dict(role='user',content=f'My email is {address}. Please look up my account.')],dict(name='find_user_id_by_email',arguments={'email':address}))
    add('name_zip_copy',[dict(role='user',content='I cannot remember my email. My name is Neris Voss and my ZIP is 46217. Please look me up.')],dict(name='find_user_id_by_name_zip',arguments=dict(first_name='Neris',last_name='Voss',zip='46217')))
    add('email_correction',intro+[call('find_user_id_by_email',{'email':email}),tool('find_user_id_by_email','Error: User not found'),
        dict(role='assistant',content='I could not find that email. Could you check the email associated with your account?'),
        dict(role='user',content=f'Sorry, that email was wrong. My account email is {new_email}. Please try this corrected email.')],dict(name='find_user_id_by_email',arguments={'email':new_email}))
    add('lookup_failure_clarify',intro+[call('find_user_id_by_email',{'email':email}),tool('find_user_id_by_email','Error: User not found')],clarify='identity')
    add('authenticated_get_profile',auth,dict(name='get_user_details',arguments={'user_id':uid}))
    add('profile_get_requested_order',profile,dict(name='get_order_details',arguments={'order_id':oid}))
    add('unknown_tool_recovery',profile+[call('get_order_by_id',{'order_id':oid}),tool('__runtime__',{'error':'Unknown tool: get_order_by_id. Use the listed get_order_details tool.'})],dict(name='get_order_details',arguments={'order_id':oid}))
    add('wrong_key_recovery',profile+[call('get_order_details',{'orderID':oid}),tool('__runtime__',{'error':"Missing required argument 'order_id'; unexpected argument 'orderID'."})],dict(name='get_order_details',arguments={'order_id':oid}))
    add('malformed_json_recovery',profile+[dict(role='assistant',content='<tool_call>{"name":"get_order_details","arguments":{"order_id":"'+oid+'"}</tool_call>'),tool('__runtime__',{'error':'invalid_tool_call_json: missing closing brace. Return a complete JSON object.'})],dict(name='get_order_details',arguments={'order_id':oid}))
    add('name_lookup_after_email_failure',intro+[call('find_user_id_by_email',{'email':email}),tool('find_user_id_by_email','Error: User not found'),
        dict(role='assistant',content='That email was not found. Please provide your first name, last name, and ZIP.'),
        dict(role='user',content='Try my name and ZIP instead: Neris Voss, 46217.')],dict(name='find_user_id_by_name_zip',arguments=dict(first_name='Neris',last_name='Voss',zip='46217')))
    add('request_confirmation_before_cancel',pending+[dict(role='assistant',content=f'Order {oid} is pending and contains a Bluetooth Speaker priced at $83.75.'),
        dict(role='user',content='I no longer need that entire order. What are the next steps to cancel it?')],clarify='confirmation')
    cancel=dict(name='cancel_pending_order',arguments=dict(order_id=oid,reason='no longer needed'))
    confirmation=dict(role='assistant',content=f'Please confirm: cancel the entire pending order {oid}, because you no longer need it. The $83.75 refund goes to your original credit card.')
    add('confirmed_execute_cancel',pending+[confirmation,dict(role='user',content='Yes, cancel the entire order because I no longer need it.')],cancel)
    add('cancel_rejected_recover',pending+[confirmation,dict(role='user',content='Yes, cancel the entire order because I no longer need it.'),
        call('cancel_pending_order',{'order_id':oid,'reason':'unwanted'}),tool('__runtime__',{'error':"reason must be 'no longer needed' or 'ordered by mistake'."})],cancel)
    add('post_cancel_no_repeat',pending+[confirmation,dict(role='user',content='Yes, cancel the entire order.'),
        call(cancel['name'],cancel['arguments']),tool(cancel['name'],{**order,'status':'cancelled','payment_history':order['payment_history']+[dict(transaction_type='refund',amount=83.75,payment_method_id='credit_card_6410287')]})],clarify='final')
    assert len(rows)==20
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
