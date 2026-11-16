"""Compare completed SFT stages on identical hand-authored diagnostic scenarios."""
import ast,gc,json,os
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.prepare_function_calling import parse_calls

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics/completed-sft-stages-four-scenarios'
STAGES=[('short-2k','outputs/sft-retail-short-2k-from-hermes/run-5c6ee49620'),('copy-repair-500','outputs/sft-retail-copy-repair-500/run-fd7056d70b'),('long-500','outputs/sft-retail-long-500-after-copy-repair/run-cbf90bb883')]

def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def fixtures():
 tree=ast.parse((ROOT/'scripts/test_short_checkpoint150_sampled.py').read_text());cases=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in node.targets):cases=ast.literal_eval(node.value)
  if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='cases' and node.value.func.attr=='append':cases.append(ast.literal_eval(node.value.args[0]))
 assert len(cases)==4;return cases

def main():
 os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
 cases=fixtures();system=json.loads((ROOT/'outputs/retail-data/short-clarification-2000/sft.jsonl').read_text().splitlines()[0])['messages'][0]
 manifest=[]
 for label,path in STAGES:
  run=ROOT/path;state=json.loads((run/'trainer_state.json').read_text());best=Path(state['best_model_checkpoint'])
  assert best.is_dir() and (best/'adapter_model.safetensors').is_file()
  manifest.append({'stage':label,'run':str(run),'checkpoint':str(best),'best_eval_loss':state['best_metric']})
 atomic(OUT/'manifest.json',{'stages':manifest,'temperature':.7,'top_p':.95,'seed':42,'max_new_tokens':128,'device':'MIG0','dtype':'bf16','note':'Identical prior hand-authored prompts and synthetic tool observation fixtures. Not full environment rollouts or benchmark rewards. Initial behavior and argument accuracy scored separately. Earlier CPU outputs may differ numerically from GPU sampling.'})
 atomic(OUT/'cases.json',cases)
 results=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
 def save(status):
  atomic(OUT/'results.json',results);atomic(OUT/'progress.json',{'status':status,'completed_cases':len(results),'target_cases':12})
  texts=[];summary=[]
  for meta in manifest:
   rows=[r for r in results if r['stage']==meta['stage']]
   text='=== '+meta['stage']+' | '+Path(meta['checkpoint']).name+' ===\n'
   for r in rows:text+='\n--- '+r['name']+' ---\nEXPECTED: '+r['expected']+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'][1:])+'\n'
   texts.append(text);(OUT/(meta['stage']+'.txt')).write_text(text)
   summary.append({'stage':meta['stage'],'checkpoint':meta['checkpoint'],'completed_cases':len(rows),'correct_identity_calls':sum(r['expected_identity_call_matched'] for r in rows),'cases':[{'name':r['name'],'initial_behavior_passed':r['initial_behavior_passed'],'identity_call_passed':r['expected_identity_call_matched'],'post_auth_passed':r.get('post_auth_get_user_matched')} for r in rows]})
  (OUT/'conversations.txt').write_text('\n\n'.join(texts));atomic(OUT/'summary.json',summary)
 def call_matches(text,expected):
  try:return parse_calls(text)==[expected]
  except ValueError:return False
 for meta in manifest:
  if sum(r['stage']==meta['stage'] for r in results)==4:continue
  save('loading '+meta['stage']);print('LOADING',meta['stage'],meta['checkpoint'],flush=True)
  policy=LocalPolicy(RetailRLConfig(initial_adapter=meta['checkpoint'],temperature=.7,top_p=.95,device='cuda'))
  def generate(messages):
   ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
   output=policy.generate({'prompt_ids':ids},128,42)
   return output['text'],output['ended_with_eos']
  for case in cases:
   if any(r['stage']==meta['stage'] and r['name']==case['name'] for r in results):continue
   messages=[system]+case.get('history',[{'role':'user','content':case.get('opening','')}])
   text,ended=generate(messages);messages.append({'role':'assistant','content':text})
   r={'stage':meta['stage'],'checkpoint':meta['checkpoint'],'name':case['name'],'expected':case['expected'],'messages':messages,'initial_output':text,'initial_ended_with_eos':ended}
   if 'followup' in case:
    if case['name'].startswith('Clarification:'):r['initial_behavior_passed']='zip' in text.lower() and 'name' not in text.lower() and 'email' not in text.lower() and '<tool_call>' not in text
    else:r['initial_behavior_passed']='email' in text.lower() and '<tool_call>' not in text
    messages.append({'role':'user','content':case['followup']});text,ended=generate(messages);messages.append({'role':'assistant','content':text});r['followup_output']=text;r['followup_ended_with_eos']=ended
   else:r['initial_behavior_passed']=call_matches(text,case['expected_call'])
   r['expected_identity_call_matched']=call_matches(text,case['expected_call'])
   if r['expected_identity_call_matched'] and 'observation' in case:
    messages.append({'role':'user','content':'<tool_response>'+json.dumps(case['observation'])+'</tool_response>'})
    text,ended=generate(messages);messages.append({'role':'assistant','content':text});r['post_auth_output']=text;r['post_auth_ended_with_eos']=ended
    r['post_auth_get_user_matched']=call_matches(text,{'name':'get_user_details','arguments':{'user_id':case['observation']['content']}})
   results.append(r);save('testing '+meta['stage']);print('CASE',meta['stage'],case['name'],{k:v for k,v in r.items() if 'passed' in k or 'matched' in k},flush=True)
  del generate,policy;gc.collect();torch.cuda.empty_cache()
 save('complete');print('DONE',OUT,flush=True)
 # Preserve the prior comparison and add the completed-stage results once.
 prior=ROOT/'outputs/diagnostics/short-checkpoint50-clarification-retry/conversations.txt'
 if prior.exists():prior.write_text(prior.read_text().split('\n\n=== COMPLETED SFT STAGES ===')[0]+'\n\n=== COMPLETED SFT STAGES ===\n'+(OUT/'conversations.txt').read_text())

if __name__=='__main__':main()
