import json,unittest
from pathlib import Path
from dataclasses import replace
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_rewards import ToolRewardTracker,shaped_reward
class NoBonusTests(unittest.TestCase):
 def setUp(self):self.cfg=load_settings('configs/rl-retail-tau2-curriculum.json');self.assertEqual(self.cfg.argument_reward_weight,0.)
 def test_original_highest_advantage_read_only_episode_gets_zero(self):
  root=Path('outputs/hf-jobs/6afa086d23985fb9cd83a823/bucket/tau2-curriculum-rl-20261115T193053/training/training/group-0000/episode-03.json');e=json.loads(root.read_text());tracker=ToolRewardTracker(self.cfg);tracker.events=e['tool_reward_audit'];score,detail=shaped_reward(e['native_reward'],e['simulation'],tracker);self.assertEqual(score,0.);self.assertEqual(detail['argument_bonus'],0.);self.assertFalse(detail['grounded_call_bonus_enabled'])
 def test_task_action_still_earns_credit(self):
  tracker=ToolRewardTracker(self.cfg);r,d=shaped_reward(0.,{'reward_info':{'db_check':{'db_reward':0},'action_checks':[{'action':{'action_id':'target'},'action_reward':1}]}},tracker);self.assertEqual(r,1.);self.assertEqual(d['tau2_sum'],1.)
 def test_penalties_remain(self):
  tracker=ToolRewardTracker(self.cfg);tracker.record('<tool_call>{bad}</tool_call>',[],[dict(role='tool',name='__runtime__',content='error',error=True)],{});r,d=shaped_reward(0.,{'reward_info':{}},tracker);self.assertLess(r,0.);self.assertGreater(d['incorrect_argument_penalty'],0.)
