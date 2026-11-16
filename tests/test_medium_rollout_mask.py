import unittest,tempfile
from pathlib import Path
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_bank import bank_environment
from tau2_grpo.medium_scenario_bank import reference_text
class RolloutTest(unittest.TestCase):
 def test_all_generated_steps_only_are_policy_actions(self):
  from transformers import AutoTokenizer
  from tau2_grpo.rl import make_episode
  cfg=load_settings('configs/rl-retail-medium.json');cid='world-0000-medium-exchange_item';env=bank_environment(cfg,cid);route=env.case['routes'][0];tok=AutoTokenizer.from_pretrained(cfg.initial_adapter)
  class P:
   tokenizer=tok;version=0;index=0
   def generate(self,prepared,max_tokens,seed):
    text=reference_text(route[self.index]);self.index+=1;tokens=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id];return dict(text=text,completion_ids=tokens,old_logprobs=[-1.]*len(tokens),ended_with_eos=True)
  with tempfile.TemporaryDirectory() as d:
   e=make_episode(P(),cfg,cid,42,Path(d)/'episode.json');self.assertTrue(e['complete'],e.get('error'));self.assertEqual(e['reward'],1.);self.assertEqual(len(e['actions']),7);self.assertEqual(e['actions'][0]['raw_message_count'],2);self.assertEqual([a['text'] for a in e['actions']],[reference_text(s) for s in route]);self.assertTrue(all(a['messages'][-1]['role'] in ('user','tool') for a in e['actions']))
