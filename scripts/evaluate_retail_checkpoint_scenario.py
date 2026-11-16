"""Evaluate a frozen SFT checkpoint on Tau2 using Transformers on CPU or GPU."""
import argparse,hashlib,json,os,shutil,time
from dataclasses import asdict
from pathlib import Path

def main():
 import torch
 from loguru import logger
 from transformers import AutoTokenizer
 from safetensors import safe_open
 from tau2_grpo.rl_config import RetailRLConfig,task_ids
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl import make_episode
 from tau2_grpo.rl_rollout import atomic_json
 from tau2_grpo.rl_reporting import write_episode_conversation
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--task-id',default='34');p.add_argument('--seed',type=int,default=100043);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=16);p.add_argument('--chat-template',default='templates/retail_aya_tool_calling.jinja');p.add_argument('--paired-first-turn',type=Path);p.add_argument('--agent-prompt-file',default='templates/glaive_decision_rules.txt');p.add_argument('--max-actions',type=int,default=40);p.add_argument('--glaive-system-format',action='store_true');p.add_argument('--transcript-file',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-head-memory-mib',type=int,default=0);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--top-p',type=float,default=1.);p.add_argument('--user-prompt-file',default='');p.add_argument('--system-prompt-file',type=Path);p.add_argument('--auto-gpu-offload',action='store_true');a=p.parse_args()
 if a.max_actions < 1: p.error('--max-actions must be positive')
 if a.temperature < 0 or not 0 < a.top_p <= 1: p.error('Invalid sampling settings')
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
 logger.remove();logger.add(__import__('sys').stderr,level='ERROR')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'episode.json').exists():raise RuntimeError('Evaluation output already exists')
 source=a.checkpoint.resolve();checkpoint_dir=source.parent if source.name=='policy' else source;step=int(checkpoint_dir.name.split('-')[-1])
 if (source/'trainer_state.json').exists():assert json.loads((source/'trainer_state.json').read_text())['global_step']==step
 else:assert json.loads((checkpoint_dir.parent/'result.json').read_text())['optimizer_steps']>=step
 with safe_open(source/'adapter_model.safetensors',framework='pt',device='cpu') as f:assert len(f.keys())>0
 snapshot=out/'checkpoint_snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['adapter_config.json','adapter_model.safetensors','chat_template.jinja','tokenizer_config.json','tokenizer.json','special_tokens_map.json','trainer_state.json']:
  if (source/name).exists():shutil.copy2(source/name,snapshot/name)
 cfg=RetailRLConfig(initial_adapter=str(snapshot),device=a.device,temperature=a.temperature,top_p=a.top_p,group_size=1,agent_prompt_file=a.agent_prompt_file,user_prompt_file=a.user_prompt_file,max_actions=a.max_actions)
 assert a.task_id in task_ids(cfg,'validation'),'Use a reserved validation task'
 atomic_json(out/'config.json',{'settings':asdict(cfg),'checkpoint_source':str(source),'adapter_sha256':hashlib.sha256((snapshot/'adapter_model.safetensors').read_bytes()).hexdigest(),'task_id':a.task_id,'seed':a.seed,'effective_chat_template':a.chat_template,'glaive_system_format':a.glaive_system_format,'system_prompt_file':str(a.system_prompt_file) if a.system_prompt_file else None,'streaming_transcript':str(a.transcript_file) if a.transcript_file else None,'effective_template_sha256':hashlib.sha256(Path(a.chat_template).read_bytes()).hexdigest(),'source_template_sha256':hashlib.sha256((source/'chat_template.jinja').read_bytes()).hexdigest(),'note':f'One diagnostic validation rollout. {a.device} BF16 Transformers inference, no parameter updates. Qwen on MIG2 is the customer, summarizer and assertion judge.'})
 atomic_json(out/'progress.json',{'stage':'loading_checkpoint','task_id':a.task_id})
 policy=LocalPolicy(cfg)
 if a.auto_gpu_offload:
  if a.gpu_head_memory_mib:raise ValueError('Choose automatic GPU dispatch or capped head offload')
  from tau2_grpo.inference_head_offload import enable_available_gpu_dispatch
  placement=enable_available_gpu_dispatch(policy)
  atomic_json(out/'gpu-placement.json',placement)
  print('GPU_PLACEMENT',json.dumps(placement),flush=True)
 if a.gpu_head_memory_mib:
  from tau2_grpo.inference_head_offload import enable_bounded_cuda_head
  try:placement=enable_bounded_cuda_head(policy,a.gpu_head_memory_mib)
  except RuntimeError as exc:
   if 'Insufficient free memory to preserve training headroom' not in str(exc):raise
   placement={'mode':'cpu_fallback','requested_allocator_mib':a.gpu_head_memory_mib,'reason':str(exc)}
  atomic_json(out/'gpu-placement.json',placement)
  print('GPU_PLACEMENT',json.dumps(placement),flush=True)
 policy.tokenizer=AutoTokenizer.from_pretrained(snapshot);policy.tokenizer.pad_token=policy.tokenizer.pad_token or policy.tokenizer.eos_token
 policy.tokenizer.chat_template=Path(a.chat_template).read_text()
 (out/'effective_chat_template.jinja').write_text(policy.tokenizer.chat_template)
 paired=None;paired_result=None
 if a.paired_first_turn:
  previous=json.loads(a.paired_first_turn.read_text());paired=previous['actions'][0]
  assert paired['temperature']==cfg.temperature and paired['top_p']==cfg.top_p
  assert policy.tokenizer.encode(paired['rendered_prompt'],add_special_tokens=False)==paired['prompt_ids']
  atomic_json(out/'progress.json',{'stage':'paired_first_turn_probe'})
  start=time.monotonic();paired_result=policy.generate(paired,cfg.response_tokens,previous['seed'])
  atomic_json(out/'paired-first-turn.json',{'source_episode':str(a.paired_first_turn),'identical_prompt_ids':True,'seed':previous['seed'],'temperature':cfg.temperature,'top_p':cfg.top_p,'baseline_step':step,'baseline_text':paired_result['text'],'post_sft_text':paired['text'],'baseline_generation_seconds':time.monotonic()-start})
  print('PAIRED FIRST TURN',json.dumps({'baseline':paired_result['text'],'post_sft':paired['text']}),flush=True)
 class InferenceOnly:
  tokenizer=policy.tokenizer;version=0
  def generate(self,prepared,max_tokens,seed):
   start=time.monotonic()
   if paired is not None and prepared['prompt_ids']==paired['prompt_ids'] and seed==a.seed:
    result=paired_result
   else:result=policy.generate(prepared,max_tokens,seed)
   with (out/'live-model-outputs.txt').open('a') as f:f.write(f"ASSISTANT ({time.monotonic()-start:.1f}s):\n{result['text']}\n\n")
   atomic_json(out/'progress.json',{'stage':'rollout','last_output':result['text'],'generation_seconds':time.monotonic()-start,'updated_at':time.time()})
   return result
 atomic_json(out/'progress.json',{'stage':'starting_rollout','task_id':a.task_id})
 if a.system_prompt_file:
  if a.glaive_system_format:raise ValueError('Choose original system prompt or Glaive framing, not both')
  import re
  from tau2_grpo.rl_tau2 import RetailEnvironment
  original_reset=RetailEnvironment.reset
  original_system=a.system_prompt_file.read_text()
  def original_prompt_reset(env):
   messages=original_reset(env)
   prior_catalog=re.findall(r'<tools>(.*?)</tools>',messages[0]['content'],re.S)
   replacement_catalog=re.findall(r'<tools>(.*?)</tools>',original_system,re.S)
   if len(prior_catalog)!=1 or len(replacement_catalog)!=1:raise ValueError('Expected one original and one runtime catalog')
   runtime_tools={t['function']['name']:t for t in json.loads(prior_catalog[0])}
   trained_tools={t['function']['name']:t for t in json.loads(replacement_catalog[0])}
   if any(runtime_tools.get(name)!=tool for name,tool in trained_tools.items()) or set(runtime_tools)-set(trained_tools) not in (set(),{'done'}):
    raise ValueError('Original training tools differ from runtime; only the optional runtime done helper may be absent')
   messages[0]['content']=original_system
   return messages
  RetailEnvironment.reset=original_prompt_reset
 if a.glaive_system_format:
  from tau2_grpo.rl_tau2 import RetailEnvironment
  from probe_retail_glaive_tool_format import glaive_framed_system
  original_reset=RetailEnvironment.reset
  def framed_reset(env):
   messages=original_reset(env)
   messages[0]['content']=glaive_framed_system(messages[0]['content'])
   return messages
  RetailEnvironment.reset=framed_reset
 import threading
 from tau2_grpo.rl_reporting import episode_check_report
 stopped=threading.Event()
 def refresh_transcript():
  record_path=out/'episode.json'
  if not record_path.exists():return
  record=json.loads(record_path.read_text())
  record['check_report']=episode_check_report(record)
  rendered_path=write_episode_conversation(record_path,record)
  if a.transcript_file:
   target=a.transcript_file.resolve();target.parent.mkdir(parents=True,exist_ok=True)
   temp=target.with_suffix('.tmp');temp.write_text(Path(rendered_path).read_text());temp.replace(target)
 def stream_transcript():
  while not stopped.wait(2):refresh_transcript()
 watcher=threading.Thread(target=stream_transcript,daemon=True);watcher.start()
 try:episode=make_episode(InferenceOnly(),cfg,a.task_id,a.seed,out/'episode.json')
 finally:
  stopped.set();watcher.join();refresh_transcript()
 transcript=write_episode_conversation(out/'episode.json',episode)
 summary={'task_id':a.task_id,'checkpoint_step':step,'reward':episode['reward'],'complete':episode['complete'],'actor_turns':len(episode['actions']),'stop_reason':episode.get('stop_reason'),'error':episode.get('error'),'checks':episode['check_report']['checks'],'turn_counts':episode['check_report']['turn_counts'],'transcript':str(transcript)}
 atomic_json(out/'summary.json',summary);atomic_json(out/'progress.json',{'stage':'complete' if episode['complete'] else 'failed',**summary});print('RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':main()
