"""Real Aya/PEFT optimizer smoke with explicitly synthetic contrast rewards.

This is NOT a Tau2 success score or production training run. Actual policy samples
and replay probabilities exercise one non-tied GRPO update through AlignTune.
"""
import json,os
from dataclasses import asdict
from pathlib import Path
import torch,wandb
from tau2_grpo.aligntune_runtime import make_owner
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.rl_objective import update_group

out=Path('outputs/aligntune-smoke/rl-optimizer');out.mkdir(parents=True,exist_ok=True)
cfg=RetailRLConfig(temperature=1.,top_p=1.,group_size=2,train_steps=1)
policy=LocalPolicy(cfg,trainable=True);tokenizer=policy.tokenizer
prompt=tokenizer.apply_chat_template([{'role':'system','content':'Answer briefly.'},{'role':'user','content':'Name one color.'}],tokenize=True,add_generation_prompt=True,return_dict=False)
actions=[]
for seed in range(42,74):
 result=policy.generate({'prompt_ids':prompt},3,seed)
 if actions and result['completion_ids']==actions[0]['completion_ids']:continue
 action=dict(prompt_ids=prompt,**result,policy_version=0)
 action['replay_old_logprobs']=policy.score(action).detach().cpu().tolist()
 action['reference_logprobs']=policy.score(action,reference=True).detach().cpu().tolist()
 actions.append(action)
 if len(actions)==2:break
assert len(actions)==2,'Could not sample two distinct completions'
episodes=[{'task_id':'__optimizer_smoke__','episode_id':f'smoke-{i}','policy_version':0,'complete':True,'reward':float(i),'actions':[a],'synthetic_reward_fixture':True} for i,a in enumerate(actions)]
parameters=[p for p in policy.model.parameters() if p.requires_grad];before=[p.detach().cpu().clone() for p in parameters]
optimizer=torch.optim.AdamW(parameters,lr=cfg.learning_rate)
wb=wandb.init(project='tau2-retail',mode='online',name='aligntune-real-aya-grpo-optimizer-smoke',dir=str(out),job_type='optimizer-smoke',config=asdict(cfg)|{'synthetic_rewards':True,'not_tau2_task_rewards':True})
owner=None
try:
 def execute():
  metrics=update_group(policy,optimizer,episodes,cfg);owner.step(policy.version,metrics);return metrics
 owner=make_owner(kind='turn_aware_tau2_grpo_smoke',config=cfg,run_dir=out,model=policy.model,dataset=episodes,tokenizer=tokenizer,execute=execute)
 metrics=owner.train();changed=sum(not torch.equal(old,p.detach().cpu()) for old,p in zip(before,parameters))
 assert metrics['optimizer_steps']==1 and metrics['grad_norm']>0 and changed>0
 assert all(bool(torch.isfinite(p).all()) for p in parameters)
 report={'passed':True,'synthetic_rewards_not_tau2':True,'changed_adapter_tensors':changed,'metrics':metrics,'sampled_completions':[a['text'] for a in actions],'wandb_url':wb.url}
 (out/'report.json').write_text(json.dumps(report,indent=2));wb.log(metrics|{'changed_adapter_tensors':changed});print(json.dumps(report,indent=2),flush=True)
 policy.model.save_pretrained(str(out/'smoke_adapter'),selected_adapters=['policy'])
finally:wb.finish()
