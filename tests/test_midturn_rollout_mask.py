import json,tempfile,unittest
from pathlib import Path
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_bank import bank_environment
from tau2_grpo.rl import make_episode
class PrefixMaskTests(unittest.TestCase):
 def test_supplied_prefix_is_context_not_sampled_actions(self):
  from transformers import AutoTokenizer
  cfg=load_settings('configs/rl-retail-midturn-pilot.json');cid='after-reply::world-0000-missing_zip';env=bank_environment(cfg,cid);prefix=env.reset();step=env.case['routes'][0][0]
  class Policy:
   version=0
   tokenizer=AutoTokenizer.from_pretrained('outputs/rl-initialization/clarification-diagnostic-step50')
   def generate(self,prepared,max_tokens,seed):
    text='<tool_call>'+json.dumps(dict(name=step['name'],arguments=step['arguments']),separators=(',',':'))+'</tool_call>';ids=self.tokenizer.encode(text,add_special_tokens=False)+[self.tokenizer.eos_token_id]
    return dict(text=text,completion_ids=ids,old_logprobs=[-1.]*len(ids),ended_with_eos=True)
  with tempfile.TemporaryDirectory() as d:
   r=make_episode(Policy(),cfg,cid,42,Path(d)/'episode.json');self.assertTrue(r['complete'],r.get('error'));self.assertEqual(r['reward'],1);self.assertEqual(len(r['actions']),1);self.assertEqual(r['actions'][0]['raw_message_count'],len(prefix));self.assertNotIn('Could you please',r['actions'][0]['text'])
if __name__=='__main__':unittest.main()
