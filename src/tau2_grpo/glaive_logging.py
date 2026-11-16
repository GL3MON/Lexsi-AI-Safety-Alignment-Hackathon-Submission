"""Per-rollout GRPO logging, including tied groups with zero advantages."""
import json
from tau2_grpo.rl_objective import group_advantages

COLUMNS=['group','rollout','task_id','scenario','policy_version','reward','advantage',
         'agent_turns','customer_replied','termination_reason','conversation','reward_details']


def rollout_records(episodes,group):
    advantages=group_advantages(episodes)
    records=[]
    for episode,advantage in zip(episodes,advantages,strict=True):
        episode['advantage']=advantage
        conversation='\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in episode['messages'])
        records.append([group,episode['episode_id'],episode['task_id'],episode['scenario'],
                        episode['policy_version'],episode['reward'],advantage,
                        episode['agent_turns'],episode['customer_replied'],
                        episode['termination_reason'],conversation,json.dumps(episode['reward_details'])])
    metrics={}
    for index,episode in enumerate(episodes):
        metrics[f'rollouts/advantage_{index}']=episode['advantage']
        metrics[f'rollouts/reward_{index}']=episode['reward']
    return records,metrics
