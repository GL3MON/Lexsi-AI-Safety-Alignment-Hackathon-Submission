"""Paired short-bank and native Tau2 evaluation; no optimization."""
import argparse,json,time,re
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_rollout import atomic_json

def json_content(text):
 # Remove a complete Markdown wrapper only when its entire body is valid JSON.
 # Do not extract an arbitrary object from commentary or repair malformed data.
 match=re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*",text,re.S)
 if match:
  try:json.loads(match.group(1))
  except ValueError:return text
  return match.group(1)
 return text

class LocalFrozenClient:
 """Minimal OpenAI response interface for the frozen Qwen customer and judge."""
 def __init__(self,cfg):
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer
  self.torch=torch;self.tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
  self.model=AutoModelForCausalLM.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
  self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
 def create(self,model,messages,max_tokens,temperature=0,seed=0,**kwargs):
  from openai.types.chat import ChatCompletion
  torch=self.torch;torch.manual_seed(seed)
  inputs=self.tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False).to('cuda');n=inputs['input_ids'].shape[1]
  with torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,use_cache=True,logits_to_keep=1)
  tokens=output[0,n:];eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  ended=bool(len(tokens) and int(tokens[-1]) in (eos or [self.tokenizer.eos_token_id]))
  content=self.tokenizer.decode(tokens,skip_special_tokens=True);raw_content=content
  if kwargs.get('response_format',{}).get('type')=='json_object':content=json_content(content)
  return ChatCompletion(local_raw_content=raw_content,id='local-frozen',created=int(time.time()),model=model,object='chat.completion',choices=[dict(index=0,finish_reason='stop' if ended else 'length',message=dict(role='assistant',content=content))])
 def close(self):pass

class InferencePolicy:
 def __init__(self,local):self.local=local;self.tokenizer=local.tokenizer;self.version=0
 def generate(self,*a,**kw):return self.local.generate(*a,**kw)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pilot',required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl import make_episode
 from tau2_grpo.rl_bank import validation_cases
 from tau2_grpo.rl_bank_eval import verify_episode_prompt
 from tau2_grpo.rl_reporting import write_episode_conversation
 import wandb
 cfg=replace(load_settings(a.config),temperature=0.);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 local=LocalPolicy(cfg,trainable=False);local.model.load_adapter(a.pilot,adapter_name='pilot',is_trainable=False);proxy=InferencePolicy(local);stages=[('pilot','pilot'),('after20','policy')];all_results={}
 metadata=dict(inference_only=True,temperature=0.,tasks=['69','34','63'],native_tau2_judge='frozen Qwen3.5-4B greedy Transformers; not official leaderboard judge',optimizer_updates=0)
 with wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=out.parent.name+'-paired-eval',config=metadata) as wb:
  atomic_json(out/'runtime.json',dict(wandb_url=wb.url,**metadata))
  for stage,alias in stages:
   local.model.set_adapter(alias);local.model.requires_grad_(False);local.model.eval();seen=set();results=[]
   for cid in validation_cases(cfg):
    file=out/stage/'short-retail114'/(cid+'-bank.json')
    episode=json.loads(file.read_text()) if file.exists() else {}
    if not episode.get('complete'):episode=make_episode(proxy,cfg,cid,cfg.seed,file)
    if episode['complete']:verify_episode_prompt(episode,seen)
    results.append(dict(task_id=cid,success=episode.get('short_task_success'),complete=episode['complete'],turns=len(episode['actions'])));atomic_json(file.parent/'summary.json',dict(results=results,passed=sum(bool(r['success']) for r in results),total=len(results),target=114,native_tau2_evaluation=False))
    if not episode['complete']:raise RuntimeError('Short evaluation infrastructure failure: '+str(episode.get('error')))
   all_results[stage]=dict(short_passed=sum(bool(r['success']) for r in results),short_total=len(results));atomic_json(out/'summary.json',all_results)
  # Keep the same frozen customer, actor decoding and task seed across both stages.
  native=replace(cfg,environment='tau2',bank_view='opening',bank_sampling='balanced',reward_mode='native',max_actions=20,response_tokens=256,max_episode_tokens=4096,prompt_budget=7680,user_model='qwen3.5-4b',split_manifest=a.split_manifest)
  manifest=json.loads(Path(a.split_manifest).read_text());tasks=['69','34','63'];assert set(tasks)<=set(manifest['partitions']['validation'])
  client=LocalFrozenClient(native)
  from tau2_grpo.rl_tau2 import ExternalChat,RetailEnvironment
  from tau2_grpo.rl_context import ContextController,FrozenSummarizer
  from tau2_grpo.rl_rollout import collect_episode
  for stage,alias in stages:
   local.model.set_adapter(alias);local.model.requires_grad_(False);local.model.eval();local.cfg=native;rows=[]
   for tid in tasks:
    external=ExternalChat(client,client.tokenizer,native,[]);summarizer=FrozenSummarizer(client,native.user_model,client.tokenizer,native.external_context_limit,native.summary_max_tokens)
    controller=ContextController(proxy.tokenizer,summarizer,native.context_limit,native.response_tokens,native.prompt_budget);env=RetailEnvironment(tid,native.seed,native,external)
    file=out/stage/'tau2'/('task-'+tid+'.json');episode=json.loads(file.read_text()) if file.exists() else {}
    if not episode.get('complete'):episode=collect_episode(proxy,env,controller,tid,native.seed,native,file)
    write_episode_conversation(file,episode)
    rows.append(dict(task_id=tid,reward=episode['reward'],complete=episode['complete'],actor_turns=len(episode['actions']),checks=episode.get('check_report'),stop_reason=episode.get('stop_reason'),error=episode.get('error')))
    atomic_json(file.parent/'summary.json',dict(results=rows,native_tau2_evaluation=True,judge=metadata['native_tau2_judge']));all_results[stage]['tau2']=rows;atomic_json(out/'summary.json',all_results)
    print(json.dumps(dict(event='paired_tau2_eval',stage=stage,task_id=tid,reward=episode['reward'],complete=episode['complete'])),flush=True)
    if not episode['complete']:raise RuntimeError('Tau2 evaluation infrastructure failure: '+str(episode.get('error')))
  wb.log({stage+'/short_success':data['short_passed']/data['short_total'] for stage,data in all_results.items()})
  (out/'COMPLETE').write_text('complete\n')
if __name__=='__main__':main()
