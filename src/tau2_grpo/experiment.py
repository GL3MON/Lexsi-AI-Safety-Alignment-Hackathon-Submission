"""Basic telecom solo GRPO using the AlignTune native-environment extension.

Retail uses tau2_grpo.rl, which preserves summary-conditioned turn replay.
The obsolete AgentTune closed-loop retraining/deployment service is retired.
"""
import json,os,random,re,sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from tau2_grpo.config import load_config

def main():
 root=Path(os.environ['TAU2_CHECKOUT']).resolve()
 if not (root/'src/tau2').is_dir():raise ValueError('Missing Tau2 checkout')
 cfg=load_config(Path(os.environ['TAU2_GRPO_CONFIG']))
 run=Path(os.environ['TAU2_GRPO_OUTPUT'])/('run-'+uuid4().hex[:10]);run.mkdir(parents=True)
 (run/'experiment_config.json').write_text(json.dumps(asdict(cfg),indent=2));print('Artifacts:',run,flush=True)
 from loguru import logger
 logger.remove();logger.add(sys.stderr,level='ERROR')
 from tau2.domains.telecom.environment import get_tasks_split
 from tau2_grpo import rollout
 from tau2_grpo.rollout import Tau2SoloRollout,BASE_PROMPT,terminal_task_reward,set_phase,cleanup_trainer_environments
 splits=get_tasks_split();train=sorted(splits['train']);random.Random(cfg.seed).shuffle(train)
 count=max(1,round(len(train)*cfg.validation_fraction));validation=train[:count][:cfg.validation_tasks];fit=train[count:][:cfg.fit_scan_tasks];test=sorted(splits['test'])[:cfg.final_test_tasks]
 assert set(fit).isdisjoint(validation) and set(train).isdisjoint(test)
 (run/'split_manifest.json').write_text(json.dumps({'fit_scan':fit,'validation_sample':validation,'test_sample':test},indent=2))
 tokenizer=None;template=None
 rollout.configure(cfg,run,fit,lambda:tokenizer,lambda:template)
 set_phase('cpu_smoke',fit);env=Tau2SoloRollout()
 try:
  context=env.reset(task_id=fit[0]);assert '<ticket>' in context and '<gym_tools>' in context
  phone=re.search(r'\b\d{3}-\d{3}-\d{4}\b',env._last_info['task'].ticket)
  if not phone:raise RuntimeError('CPU smoke task has no phone number')
  assert 'customer_id' in env.execute('get_customer_by_phone',json.dumps({'phone_number':phone[0]}))
  env.execute('done','{}');env._finish()
 finally:
  if env._env is not None:env._finish()
 if os.environ.get('TAU2_GRPO_SMOKE')=='1':print('CPU solo bridge passed; no model weights loaded.');return
 import torch,wandb
 from datasets import Dataset
 from peft import LoraConfig
 from transformers import AutoConfig,AutoTokenizer,set_seed
 from trl import GRPOConfig,GRPOTrainer
 from tau2_grpo.aligntune_runtime import make_owner
 from tau2_grpo.callbacks import AdapterUpdateAudit
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose one GPU')
 base=AutoConfig.from_pretrained(cfg.model_id,revision=cfg.model_revision);revision=getattr(base,'_commit_hash',None) or cfg.model_revision
 if cfg.context_limit>base.max_position_embeddings:raise ValueError('Context exceeds base model capability')
 tokenizer=AutoTokenizer.from_pretrained(cfg.model_id,revision=revision);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token;tokenizer.padding_side='left';template=tokenizer.chat_template
 set_seed(cfg.seed);dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 if os.environ.get('TAU2_GRPO_PREVIEW')=='1':
  from tau2_grpo.preview import run_prompt_preview
  run_prompt_preview(cfg,run,tokenizer,revision,dtype,fit[0]);return
 def data(ids):return Dataset.from_list([{'prompt':[{'role':'user','content':BASE_PROMPT}],'task_id':i} for i in ids])
 args=GRPOConfig(output_dir=str(run/'checkpoints'),max_steps=cfg.train_steps,per_device_train_batch_size=1,gradient_accumulation_steps=cfg.group_size,steps_per_generation=cfg.group_size,per_device_eval_batch_size=cfg.group_size,num_generations=cfg.group_size,num_generations_eval=cfg.group_size,max_completion_length=cfg.max_completion_tokens,max_tool_calling_iterations=cfg.max_tool_iterations,temperature=cfg.temperature,top_p=cfg.top_p,learning_rate=cfg.learning_rate,beta=cfg.kl_beta,bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},model_init_kwargs={'revision':revision,'dtype':dtype,'attn_implementation':'sdpa','use_cache':False},generation_kwargs={'use_cache':True},chat_template_kwargs={'enable_thinking':False},use_vllm=False,remove_unused_columns=False,mask_truncated_completions=False,loss_type='grpo',scale_rewards='group',logging_steps=1,save_steps=max(1,cfg.train_steps//2),save_total_limit=2,eval_strategy='no',report_to=['wandb'],seed=cfg.seed,data_seed=cfg.seed)
 lora=LoraConfig(r=cfg.lora_rank,lora_alpha=2*cfg.lora_rank,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM')
 wb=wandb.init(project='tau2-retail',mode='online',name=run.name,dir=str(run),config=asdict(cfg)|{'framework':'aligntune','domain':'telecom'})
 trainer=None;success=False
 try:
  audit=AdapterUpdateAudit();trainer=GRPOTrainer(model=cfg.model_id,args=args,processing_class=tokenizer,peft_config=lora,train_dataset=data(fit),reward_funcs=terminal_task_reward,environment_factory=Tau2SoloRollout,callbacks=[audit]);template=trainer.chat_template
  owner=make_owner(kind='telecom_native_environment_grpo',config=cfg,run_dir=run,model=trainer.model,dataset=fit,tokenizer=tokenizer,execute=trainer.train)
  for callback in owner.get_hf_callbacks():trainer.add_callback(callback)
  set_phase('train',fit);result=owner.train();trainer.save_model(str(run/'candidate_adapter'));tokenizer.save_pretrained(run/'candidate_adapter')
  metrics=dict(result.metrics)
  set_phase('validation',validation);metrics['validation']=trainer.evaluate(eval_dataset=data(validation),metric_key_prefix='validation')
  set_phase('test',test);metrics['test']=trainer.evaluate(eval_dataset=data(test),metric_key_prefix='test')
  (run/'training_metrics.json').write_text(json.dumps(metrics,indent=2));success=True
 finally:
  if trainer is not None:cleanup_trainer_environments(trainer)
  wb.finish(exit_code=0 if success else 1)

if __name__=='__main__':main()
