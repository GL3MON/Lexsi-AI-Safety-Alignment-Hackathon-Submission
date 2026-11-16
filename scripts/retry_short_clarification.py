"""Fresh diagnostic prompts and greedy outputs; no training or task reward."""
import json,re,os
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.prepare_function_calling import parse_calls

torch.set_num_threads(8)
torch.set_num_interop_threads(2)
out=Path('outputs/diagnostics/short-checkpoint50-clarification-retry');out.mkdir(parents=True,exist_ok=True)
adapter='outputs/sft-retail-short-2k-from-hermes/run-5c6ee49620/checkpoint-50'
policy=LocalPolicy(RetailRLConfig(initial_adapter=adapter,temperature=0,device='cpu'))
system=json.loads(Path('outputs/retail-data/short-clarification-2000/sft.jsonl').read_text().splitlines()[0])['messages'][0]
cases=[
 {'name':'Missing identity, then email supplied','opening':"Hi, I accidentally ordered a desk fan and want to cancel it. I cannot find the order number.",'expected':'Ask for missing identity; do not guess an identifier.','followup':"My email is dalia.moreno.eval@example.org.",'expected_call':{'name':'find_user_id_by_email','arguments':{'email':'dalia.moreno.eval@example.org'}}},
 {'name':'Name and ZIP already supplied','opening':"I'd like to return my delivered headphones to the original payment method. I cannot remember my email. My first name is Jonas, my last name is Keller, and my ZIP code is 60617. I don't know the order number.",'expected':'Immediately look up Jonas Keller using ZIP 60617, without another identity question.','expected_call':{'name':'find_user_id_by_name_zip','arguments':{'first_name':'Jonas','last_name':'Keller','zip':'60617'}},'observation':{'name':'find_user_id_by_name_zip','content':'jonas_keller_eval_72'}},
 {'name':'Failed email lookup, then full alternative identity supplied','history':[{'role':'user','content':"I need to cancel an order I no longer need. My email is ravi.dutta.typo@example.org."},{'role':'assistant','content':'I will check your account using that email.\n<tool_call>{"name":"find_user_id_by_email","arguments":{"email":"ravi.dutta.typo@example.org"}}</tool_call>'},{'role':'user','content':'<tool_response>{"name":"find_user_id_by_email","content":{"error":"User not found"}}</tool_response>'},{'role':'assistant','content':'That email did not match an account. Could you provide your first name, last name, and ZIP code?'},{'role':'user','content':"That may be an old email. Please use my name instead: first name Ravi, last name Dutta, ZIP code 02139."}],'expected':'Use the supplied name and ZIP, preserving the leading zero. Do not retry the failed email or ask for known details.','expected_call':{'name':'find_user_id_by_name_zip','arguments':{'first_name':'Ravi','last_name':'Dutta','zip':'02139'}},'observation':{'name':'find_user_id_by_name_zip','content':'ravi_dutta_eval_84'}}]
cases.append({'name':'Clarification: full name known, ZIP missing','opening':"I want to return my delivered desk lamp. I do not remember my email address. My first name is Maya and my last name is Ellis. I do not know the order number.",'expected':'Ask only for the missing ZIP code, then use Maya Ellis and the supplied ZIP. Do not ask for email or name again.','followup':"My ZIP code is 04861.",'expected_call':{'name':'find_user_id_by_name_zip','arguments':{'first_name':'Maya','last_name':'Ellis','zip':'04861'}}})
cases=cases[-1:]
results=json.loads((out/'results.json').read_text())['cases'] if (out/'results.json').exists() else []
def generate(messages):
 ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
 result=policy.generate({'prompt_ids':ids},128,42)
 return result['text'],len(ids),result['ended_with_eos']
def save():
 (out/'results.json').write_text(json.dumps({'adapter':adapter,'device':'cpu','precision':'bf16','temperature':0,'max_new_tokens':128,'note':'Fresh hand-authored prompts with synthetic tool observation fixtures; not Tau2 reward evaluation. Follow-up after tool result is probed only when the model makes the expected originating call.','cases':results},indent=2))
 texts=[]
 for r in results:
  text='=== '+r['name']+' ===\nEXPECTED: '+r['expected']+'\n\n'
  for m in r['messages'][1:]:text+=('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content']+'\n\n'
  texts.append(text)
 (out/'conversations.txt').write_text('\n'.join(texts))
for case in cases:
 if any(r['name']==case['name'] for r in results):continue
 messages=[system]+case.get('history',[{'role':'user','content':case.get('opening','')}])
 text,tokens,ended=generate(messages);messages.append({'role':'assistant','content':text})
 result={'name':case['name'],'expected':case['expected'],'messages':messages,'initial_prompt_tokens':tokens,'initial_ended_with_eos':ended,'initial_output':text}
 print(case['name'],text,flush=True)
 if 'followup' in case:
  messages.append({'role':'user','content':case['followup']});text,tokens,ended=generate(messages);messages.append({'role':'assistant','content':text})
  result['followup_output']=text
 try:matched=parse_calls(text)==[case['expected_call']]
 except ValueError:matched=False
 result['expected_identity_call_matched']=matched
 if matched and 'observation' in case:
  messages.append({'role':'user','content':'<tool_response>'+json.dumps(case['observation'])+'</tool_response>'})
  text,_,_=generate(messages);messages.append({'role':'assistant','content':text});result['post_auth_output']=text
  try:result['post_auth_get_user_matched']=parse_calls(text)==[{'name':'get_user_details','arguments':{'user_id':case['observation']['content']}}]
  except ValueError:result['post_auth_get_user_matched']=False
 results.append(result);save()
 print('CASE DONE',case['name'],json.dumps({k:v for k,v in result.items() if 'matched' in k}),flush=True)
print('DONE',out,flush=True)
