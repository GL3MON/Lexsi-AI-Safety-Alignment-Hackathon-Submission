"""Exact-prompt, completion-only DPO using AlignTune's explicit trainer lifecycle."""
import argparse,hashlib,json,math,random,shutil
from dataclasses import dataclass,asdict,fields,replace
from pathlib import Path
from types import SimpleNamespace

@dataclass
class DPOSettings:
 model_id:str='outputs/hf-export/tiny-aya-global-function-calling-sft-525'
 initial_adapter:str='outputs/rl-initialization/post-rl-clarification-sft-best19'
 dataset_path:str='outputs/retail-data/clarification-dpo/train.jsonl'
 eval_dataset_path:str='outputs/retail-data/clarification-dpo/validation.jsonl'
 output_dir:str='outputs/dpo-clarification'
 chat_template_file:str='templates/retail_sdg_v2.jinja'
 beta:float=.1
 learning_rate:float=5e-6
 epochs:int=2
 pairs_per_batch:int=2
 gradient_accumulation_steps:int=8
 eval_steps:int=5
 score_chunk_size:int=16
 max_grad_norm:float=1.
 seed:int=42
 device:str='cuda'
 precision:str='bf16'
 wandb_project:str='tau2-retail'
 wandb_mode:str='online'

def load_settings(filename):
 data=json.loads(Path(filename).read_text());allowed={f.name for f in fields(DPOSettings)}
 if set(data)-allowed:raise ValueError('Unknown DPO settings')
 cfg=DPOSettings(**data)
 for k in ['epochs','pairs_per_batch','gradient_accumulation_steps','eval_steps','score_chunk_size']:
  if type(getattr(cfg,k)) is not int or getattr(cfg,k)<1:raise ValueError('Invalid '+k)
 for k in ['beta','learning_rate','max_grad_norm']:
  if not math.isfinite(getattr(cfg,k)) or getattr(cfg,k)<=0:raise ValueError('Invalid '+k)
 if cfg.wandb_mode not in ('online','offline'):raise ValueError('W&B required')
 return cfg

def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta):
 import torch.nn.functional as F
 logits=beta*((chosen-rejected)-(reference_chosen.detach()-reference_rejected.detach()))
 return F.softplus(-logits),logits

def read_pairs(filename):
 rows=[json.loads(l) for l in Path(filename).read_text().splitlines() if l.strip()]
 if not rows:raise ValueError('Empty preferences')
 seen=set()
 for r in rows:
  if r['id'] in seen:raise ValueError('Duplicate preference id')
  seen.add(r['id'])
  if not r['prompt_ids'] or not r['chosen_ids'] or not r['rejected_ids'] or r['chosen_ids']==r['rejected_ids']:raise ValueError('Invalid preference tokens')
  if len(r['prompt_ids'])+max(len(r['chosen_ids']),len(r['rejected_ids']))>8192:raise ValueError('Preference exceeds8K')
 return rows

def actions(rows):
 return [dict(prompt_ids=r['prompt_ids'],completion_ids=r[key]) for r in rows for key in ('chosen_ids','rejected_ids')]

def cache_reference(policy,rows,batch_size):
 cache={}
 for start in range(0,len(rows),batch_size):
  batch=rows[start:start+batch_size];scores=policy.score_batch(actions(batch),reference=True)
  for i,r in enumerate(batch):cache[r['id']]=[float(scores[2*i].sum()),float(scores[2*i+1].sum())]
  if start%20==0:print(json.dumps(dict(event='dpo_reference_progress',completed=min(start+batch_size,len(rows)),total=len(rows))),flush=True)
 return cache

def validate(policy,rows,reference,cfg):
 import torch
 totals=[];by_category={}
 for start in range(0,len(rows),cfg.pairs_per_batch):
  batch=rows[start:start+cfg.pairs_per_batch];scores=policy.score_batch(actions(batch))
  chosen=torch.stack([scores[2*i].sum() for i in range(len(batch))]);rejected=torch.stack([scores[2*i+1].sum() for i in range(len(batch))]);ref=torch.tensor([reference[r['id']] for r in batch],device=chosen.device)
  losses,margins=dpo_loss(chosen,rejected,ref[:,0],ref[:,1],cfg.beta)
  for i,r in enumerate(batch):
   item=dict(loss=float(losses[i]),margin=float(margins[i]),correct=float(margins[i]>0),raw_correct=float(chosen[i]>rejected[i]));totals.append(item);by_category.setdefault(r['category'],[]).append(item)
 def mean(items):return {k:sum(r[k] for r in items)/len(items) for k in ('loss','margin','correct','raw_correct')}
 return dict(**mean(totals),categories={k:mean(v) for k,v in by_category.items()},pairs=len(rows))

def train(cfg):
 import torch,wandb
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_config import RetailRLConfig
 from tau2_grpo.rl_rollout import atomic_json
 from tau2_grpo.aligntune_runtime import make_owner,verify_aligntune
 verify_aligntune()
 train_rows=read_pairs(cfg.dataset_path);eval_rows=read_pairs(cfg.eval_dataset_path)
 if {r['world_id'] for r in train_rows}&{r['world_id'] for r in eval_rows}:raise ValueError('Preference worlds overlap')
 out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(cfg.seed)
 policy_cfg=RetailRLConfig(model_id=cfg.model_id,initial_adapter=cfg.initial_adapter,temperature=1.,device=cfg.device,precision=cfg.precision,score_chunk_size=cfg.score_chunk_size,chat_template_file=cfg.chat_template_file)
 policy=LocalPolicy(policy_cfg,trainable=True)
 # Independently re-render every pair: saved IDs may never hide a template mismatch.
 from tau2_grpo.dpo_data import render_completion
 for index,r in enumerate(train_rows+eval_rows):
  for text,key in [(r['chosen'],'chosen_ids'),(r['rejected'],'rejected_ids')]:
   p,c=render_completion(policy.tokenizer,r['prompt'],text)
   if p!=r['prompt_ids'] or c!=r[key]:raise ValueError('Preference prompt/token mismatch: '+r['id'])
  if (index+1)%50==0:print('DPO_PROMPT_VALIDATION',index+1,len(train_rows+eval_rows),flush=True)
 with wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=out.name,config=asdict(cfg)) as wb:
  atomic_json(out/'runtime.json',dict(wandb_url=wb.url,algorithm='DPO',temperature_for_logprobs=1.,reference='Frozen SFT19, not adapter-disabled base',prompt_tokens_in_loss=False))
  print('DPO_REFERENCE_CACHE_START',flush=True)
  reference=cache_reference(policy,train_rows+eval_rows,cfg.pairs_per_batch)
  atomic_json(out/'reference-logps.json',dict(adapter_sha256=hashlib.sha256((Path(cfg.initial_adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),train_sha256=hashlib.sha256(Path(cfg.dataset_path).read_bytes()).hexdigest(),eval_sha256=hashlib.sha256(Path(cfg.eval_dataset_path).read_bytes()).hexdigest(),scores=reference))
  baseline=validate(policy,eval_rows,reference,cfg);atomic_json(out/'baseline-preference-eval.json',baseline)
  params=[p for n,p in policy.model.named_parameters() if p.requires_grad and '.reference.' not in n];optimizer=torch.optim.AdamW(params,lr=cfg.learning_rate)
  total=math.ceil(math.ceil(len(train_rows)/cfg.pairs_per_batch)/cfg.gradient_accumulation_steps)*cfg.epochs
  best=[float('inf'),None];history=[];rng=random.Random(cfg.seed);owner=None
  def evaluate_and_save(step):
   metrics=validate(policy,eval_rows,reference,cfg);checkpoint=out/f'checkpoint-{step}'
   policy.model.save_pretrained(checkpoint,selected_adapters=['policy']);policy.tokenizer.save_pretrained(checkpoint/'policy')
   torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),python_rng=rng.getstate()),checkpoint/'optimizer.pt')
   atomic_json(checkpoint/'state.json',dict(step=step,eval=metrics));(checkpoint/'COMPLETE').write_text('complete\n')
   if metrics['loss']<best[0]:best[:]=[metrics['loss'],checkpoint]
   atomic_json(out/'selection.json',dict(best_checkpoint=str(best[1]),best_preference_loss=best[0],latest_checkpoint=str(checkpoint)))
   all_checkpoints=sorted((p for p in out.glob('checkpoint-*') if (p/'COMPLETE').exists()),key=lambda p:int(p.name.split('-')[-1]));keep={checkpoint,best[1]}
   for p in reversed(all_checkpoints):
    if len(keep)>=2:break
    keep.add(p)
   for p in all_checkpoints:
    if p not in keep:shutil.rmtree(p)
   wb.log({'eval/loss':metrics['loss'],'eval/reference_adjusted_preference_accuracy':metrics['correct'],'eval/raw_preference_accuracy':metrics['raw_correct'],'optimizer_step':step})
   print(json.dumps(dict(event='dpo_preference_eval',step=step,**metrics)),flush=True)
  def execute():
   print('DPO_TRAIN_START',total,flush=True)
   step=0
   for epoch in range(cfg.epochs):
    rows=list(train_rows);rng.shuffle(rows);batches=[rows[i:i+cfg.pairs_per_batch] for i in range(0,len(rows),cfg.pairs_per_batch)]
    for start in range(0,len(batches),cfg.gradient_accumulation_steps):
     bucket=batches[start:start+cfg.gradient_accumulation_steps];pair_count=sum(len(b) for b in bucket);optimizer.zero_grad(set_to_none=True);loss_sum=0.;wins=0
     for batch in bucket:
      scores=policy.score_batch(actions(batch),grad=True);chosen=torch.stack([scores[2*i].sum() for i in range(len(batch))]);rejected=torch.stack([scores[2*i+1].sum() for i in range(len(batch))]);ref=torch.tensor([reference[r['id']] for r in batch],device=chosen.device)
      losses,margins=dpo_loss(chosen,rejected,ref[:,0],ref[:,1],cfg.beta);(losses.sum()/pair_count).backward();loss_sum+=float(losses.detach().sum());wins+=int((margins.detach()>0).sum())
     norm=torch.nn.utils.clip_grad_norm_(params,cfg.max_grad_norm)
     if not torch.isfinite(norm):raise RuntimeError('Nonfinite DPO gradient')
     optimizer.step();step+=1
     metrics=dict(step=step,epoch=epoch+1,loss=loss_sum/pair_count,reference_adjusted_preference_accuracy=wins/pair_count,grad_norm=float(norm),pairs=pair_count,total_steps=total);history.append(metrics);atomic_json(out/'progress.json',dict(history=history,step=step,target_steps=total));wb.log(metrics);print(json.dumps(dict(event='dpo_update',**metrics)),flush=True)
     if owner:owner.step(step,metrics)
     if step%cfg.eval_steps==0 or step==total:evaluate_and_save(step)
   shutil.copytree(best[1]/'policy',out/'final_adapter',dirs_exist_ok=True)
   atomic_json(out/'result.json',dict(complete=True,optimizer_steps=step,selected_checkpoint=str(best[1]),best_preference_loss=best[0],initial_adapter=cfg.initial_adapter,final_adapter=str(out/'final_adapter')))
   (out/'DPO_COMPLETE').write_text('complete\n');return SimpleNamespace(global_step=step)
  owner=make_owner(kind='exact_prompt_completion_dpo',config=cfg,run_dir=out,model=policy.model,dataset=train_rows,tokenizer=policy.tokenizer,execute=execute);owner.train()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();cfg=load_settings(a.config)
 if a.check:
  train_rows=read_pairs(cfg.dataset_path);eval_rows=read_pairs(cfg.eval_dataset_path);assert not {r['world_id'] for r in train_rows}&{r['world_id'] for r in eval_rows};print(json.dumps(dict(ready=True,train_pairs=len(train_rows),eval_pairs=len(eval_rows),beta=cfg.beta,reference='Frozen initializer',training_started=False)));return
 train(cfg)
if __name__=='__main__':main()
