import json,random,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from transformers import AutoTokenizer
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_bank import load_bank,balanced_order,validation_cases
from tau2_grpo.rl import make_episode

class BankIntegrationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=load_settings('hf_jobs/short_bank_rl/config.json')
  cls.root,cls.rows,cls.pools,_=load_bank(cls.cfg.scenario_bank)
 def test_balanced_sampler_and_partition(self):
  ids=task_ids(self.cfg);self.assertEqual(len(ids),2400)
  order=balanced_order(self.cfg,ids,random.Random(42))
  self.assertEqual(len(set(order)),2400)
  for i in range(0,2400,30):self.assertEqual(len({self.rows[t]['family'] for t in order[i:i+30]}),30)
  self.assertFalse(set(ids)&set(task_ids(self.cfg,'validation')))
 def test_one_validation_case_per_family(self):
  cases=validation_cases(self.cfg);self.assertEqual(len(cases),30)
  self.assertTrue(set(cases)<=set(task_ids(self.cfg,'validation')))
 def test_actual_collector_with_reference_policy(self):
  tokenizer=AutoTokenizer.from_pretrained('outputs/rl-initialization/focused-clarification200-checkpoint-25')
  class ReferencePolicy:
   version=0
   def __init__(self,route):self.tokenizer=tokenizer;self.route=iter(route)
   def generate(self,prepared,max_tokens,seed):
    s=next(self.route)
    if s['kind']=='call':text='<tool_call>'+json.dumps(dict(name=s['name'],arguments=s['arguments']),separators=(',',':'))+'</tool_call>'
    elif s['kind']=='ask':text='Could you please provide your '+{'email':'correct email address','first_name':'first name','last_name':'last name','zip':'ZIP code'}[s['field']]+'?'
    elif s['kind']=='confirm':text='Please confirm: shall I cancel the entire order?'
    elif s['kind']=='decline':text='I will leave everything unchanged.'
    else:text=f'Order {s["order_id"]} is {s["status"]}.'
    tokens=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
    assert len(tokens)<=max_tokens
    return dict(text=text,completion_ids=tokens,old_logprobs=[-1.]*len(tokens),ended_with_eos=True)
  with tempfile.TemporaryDirectory() as d:
   for cid in validation_cases(self.cfg):
    with self.subTest(case=cid):
     episode=make_episode(ReferencePolicy(self.rows[cid]['routes'][0]),self.cfg,cid,42,Path(d)/(cid+'.json'))
     self.assertTrue(episode['complete'],episode.get('error'));self.assertEqual(episode['reward'],1.)
     self.assertIsNone(episode['native_reward']);self.assertTrue(episode['short_task_success'])
 def test_incorrect_action_is_not_positive(self):
  from tau2_grpo.rl_bank import bank_environment
  env=bank_environment(self.cfg,task_ids(self.cfg)[0]);env.reset();env.step('I have done everything successfully.')
  self.assertEqual(env.finish()[0],-1.)
if __name__=='__main__':unittest.main()
