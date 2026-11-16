#!/usr/bin/env python3
"""Actual AlignTune PEFT merges of cp150 at four strengths, then identical probes."""
import ast,gc,json,math,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/deps/aligntune'
sys.path.insert(0,str(SOURCE/'src'))
OUT=ROOT/'outputs/diagnostics/aligntune-checkpoint150-strength-sweep'

def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def load_cases():
 tree=ast.parse((ROOT/'scripts/test_short_checkpoint150_sampled.py').read_text())
 cases=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in node.targets):cases=ast.literal_eval(node.value)
  if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='cases' and node.value.func.attr=='append':cases.append(ast.literal_eval(node.value.args[0]))
 assert len(cases)==4
 return cases

def main():
 import torch
 from peft.tuners.lora.layer import LoraLayer
 from aligntune.core.merge import PEFTMerger
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_config import RetailRLConfig
 from tau2_grpo.prepare_function_calling import parse_calls
 os.chdir(ROOT);torch.set_num_threads(8);torch.set_num_interop_threads(2)
 OUT.mkdir(parents=True,exist_ok=True)
 snapshot=OUT/'source_adapter'
 if not snapshot.exists():
  shutil.copytree(ROOT/'outputs/sft-retail-short-2k-from-hermes/run-5c6ee49620/checkpoint-150',snapshot,ignore=shutil.ignore_patterns('optimizer.pt','scheduler.pt','rng_state*','trainer_state.json','training_args.bin'))
 manifest={'backend':'AlignTune PEFTMerger.merge_lora','aligntune_revision':(SOURCE/'.git/HEAD').read_text().strip(),'formula':'Hermes-SFT merged base + strength * checkpoint150 retail delta','strengths':[.25,.5,.75,1.0],'temperature':.7,'top_p':.95,'seed':42,'precision':'bf16','device':'cpu','max_new_tokens':128,'base_model':'outputs/hf-export/tiny-aya-global-function-calling-sft-525','snapshot':str(snapshot),'note':'Actual BF16 model merges. Small rounding differences from runtime adapter scaling are possible. Hand-authored prompts; synthetic tool observation fixtures; not Tau2 rewards.'}
 # Resolve exact checkout commit without logging credentials.
 import subprocess
 manifest['aligntune_revision']=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
 atomic(OUT/'manifest.json',manifest)
 cases=load_cases();system=json.loads((ROOT/'outputs/retail-data/short-clarification-2000/sft.jsonl').read_text().splitlines()[0])['messages'][0]
 results= json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
 def save(status):
  atomic(OUT/'results.json',results)
  atomic(OUT/'progress.json',{'status':status,'completed_cases':len(results),'target_cases':16,'updated_at':time.time()})
  text=[]
  for result in results:
   text.append(f"=== ADAPTER STRENGTH {result['strength']} | {result['name']} ===\nEXPECTED: {result['expected']}\n\n"+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in result['messages'][1:]))
  (OUT/'conversations.txt').write_text('\n\n'.join(text))
  summary=[]
  for a in manifest['strengths']:
   rows=[r for r in results if r['strength']==a]
   summary.append({'strength':a,'completed_cases':len(rows),'correct_identity_calls':sum(r['expected_identity_call_matched'] for r in rows),'initial_clarification_checks':[{'name':r['name'],'passed':r.get('initial_clarification_check')} for r in rows if 'initial_clarification_check' in r],'post_authentication_checks':[{'name':r['name'],'passed':r['post_auth_get_user_matched']} for r in rows if 'post_auth_get_user_matched' in r]})
  atomic(OUT/'summary.json',summary)
 for strength in manifest['strengths']:
  if sum(r['strength']==strength for r in results)==4:continue
  save('merging strength '+str(strength))
  print('MERGE START',strength,flush=True)
  policy=LocalPolicy(RetailRLConfig(initial_adapter=str(snapshot),temperature=.7,top_p=.95,device='cpu'))
  layers=0
  for layer in policy.model.modules():
   if isinstance(layer,LoraLayer) and 'policy' in layer.scaling:
    layer.set_scale('policy',strength);layers+=1
  assert layers==252,('Unexpected adapted layer count',layers)
  merged_path=OUT/f'merged-strength-{strength}'
  native=policy.model.get_base_model()
  PEFTMerger().merge_lora(base_model=policy.model,output_path=str(merged_path),tokenizer=policy.tokenizer,provenance={'tool':'AlignTune','revision':manifest['aligntune_revision'],'strength':strength,'source_checkpoint':str(snapshot),'formula':manifest['formula']})
  # merge_and_unload modifies these same native layers; retain the merged native model.
  policy.model=native;policy.model.eval()
  assert not any(isinstance(layer,LoraLayer) for layer in policy.model.modules())
  print('MERGE SAVED',strength,str(merged_path),flush=True)
  def generate(messages):
   ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
   result=policy.generate({'prompt_ids':ids},128,42)
   return result['text'],result['ended_with_eos']
  def matches(text,call):
   try:return parse_calls(text)==[call]
   except ValueError:return False
  for case in cases:
   if any(r['strength']==strength and r['name']==case['name'] for r in results):continue
   messages=[system]+case.get('history',[{'role':'user','content':case.get('opening','')}])
   text,ended=generate(messages);messages.append({'role':'assistant','content':text})
   r={'strength':strength,'name':case['name'],'expected':case['expected'],'messages':messages,'initial_output':text,'initial_ended_with_eos':ended,'merged_model':str(merged_path)}
   if 'followup' in case:
    if case['name'].startswith('Clarification:'):r['initial_clarification_check']='zip' in text.lower() and 'name' not in text.lower() and 'email' not in text.lower() and '<tool_call>' not in text
    else:r['initial_clarification_check']='email' in text.lower() and '<tool_call>' not in text
    messages.append({'role':'user','content':case['followup']});text,ended=generate(messages);messages.append({'role':'assistant','content':text});r['followup_output']=text;r['followup_ended_with_eos']=ended
   r['expected_identity_call_matched']=matches(text,case['expected_call'])
   if r['expected_identity_call_matched'] and 'observation' in case:
    messages.append({'role':'user','content':'<tool_response>'+json.dumps(case['observation'])+'</tool_response>'})
    text,ended=generate(messages);messages.append({'role':'assistant','content':text});r['post_auth_output']=text;r['post_auth_ended_with_eos']=ended
    r['post_auth_get_user_matched']=matches(text,{'name':'get_user_details','arguments':{'user_id':case['observation']['content']}})
   results.append(r);save('testing strength '+str(strength))
   print('CASE DONE',strength,case['name'],{k:v for k,v in r.items() if 'check' in k or 'matched' in k},flush=True)
  del generate,native,policy;gc.collect()
 save('complete');print('DONE',OUT,flush=True)

if __name__=='__main__':main()
