"""One unassisted native Tau2 rollout after SFT, then exit without further work."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_rollout import collect_episode,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--task-id',default='69');a=p.parse_args();cfg=replace(load_settings(a.config),assistance_level='none',assisted_adaptive=False,reward_mode='native',temperature=0.,max_actions=20,max_episode_tokens=5120,response_tokens=256)
 assert a.task_id in task_ids(cfg,'validation');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2_grpo.rl_post20_eval import LocalFrozenClient,InferencePolicy
 from tau2_grpo.rl_tau2 import ExternalChat,RetailEnvironment
 from tau2_grpo.rl_context import ContextController,FrozenSummarizer
 from tau2_grpo.rl_reporting import write_episode_conversation
 import wandb
 policy=LocalPolicy(cfg,trainable=False);client=LocalFrozenClient(cfg);external=ExternalChat(client,client.tokenizer,cfg,[]);memory=FrozenSummarizer(client,cfg.user_model,client.tokenizer,cfg.external_context_limit,cfg.summary_max_tokens);controller=ContextController(policy.tokenizer,memory,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget);env=RetailEnvironment(a.task_id,cfg.seed,cfg,external)
 with wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=out.parent.name+'-post-sft-rollout',config=dict(task_id=a.task_id,unassisted=True,temperature=0.,training=False,initial_adapter=cfg.initial_adapter,max_actions=20)) as wb:
  atomic_json(out/'runtime.json',dict(wandb_url=wb.url));file=out/('task-'+a.task_id+'.json');episode=collect_episode(InferencePolicy(policy),env,controller,a.task_id,cfg.seed,cfg,file);txt=write_episode_conversation(file,episode)
  result=dict(task_id=a.task_id,unassisted=True,native_tau2_evaluation=True,native_reward=episode.get('native_reward'),complete=episode['complete'],actor_turns=len(episode['actions']),stop_reason=episode.get('stop_reason'),error=episode.get('error'),checks=episode.get('check_report'),transcript=str(txt),note='One greedy development rollout with frozen Qwen customer/summary/judge, not an official benchmark score.')
  atomic_json(out/'summary.json',result);wb.log(dict(native_reward=episode.get('native_reward'),complete=episode['complete'],actor_turns=len(episode['actions']),conversation=wandb.Table(columns=['role','content'],data=[[m['role'],m['content']] for m in episode['messages'] if m['role']!='system'])));print('POST_SFT_ROLLOUT_RESULT',json.dumps(result),flush=True)
  # This is the last action in the job: never chain RL or further training.
  (out/'ROLLOUT_FINISHED').write_text('complete\n' if episode['complete'] else 'unscored infrastructure failure; see summary.json\n')
if __name__=='__main__':main()
