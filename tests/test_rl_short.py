import json,unittest
from dataclasses import replace
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_short import ShortRetailEnvironment,KINDS,clarification_ok
class ShortTests(unittest.TestCase):
 def cfg(self,kind):return replace(RetailRLConfig(),environment='short_retail',reward_mode='short_verifiable',short_scenario=kind,system_prompt_file='templates/retail_apigen_sft_system.txt')
 def test_all_correct_short_paths_use_real_tools_and_finish_within_four_turns(self):
  for kind in KINDS:
   with self.subTest(kind=kind):
    env=ShortRetailEnvironment('113',42,self.cfg(kind));messages=env.reset();self.assertTrue(messages)
    for expected in env.expected:
     if isinstance(expected,dict):text='<tool_call>'+json.dumps(expected)+'</tool_call>'
     elif expected=='report_status':text=f'Order {env.order_id} is {env.order.status}.'
     else:text={'ask:zip':'Could you please provide your ZIP code?','ask:first_name':'Could you please provide your first name?','ask:last_name':'Could you please provide your last name?','ask:email':'Could you please confirm the correct email address?'}[expected]
     incoming,done,_=env.step(text)
    reward,simulation=env.finish();self.assertTrue(done);self.assertEqual(reward,1);self.assertTrue(simulation['short_success']);self.assertLessEqual(len(env.checks),4)
 def test_filler_is_failure_in_call_required_state(self):
  env=ShortRetailEnvironment('113',42,self.cfg('email_lookup'));env.reset();env.step('I will find your account now.');self.assertEqual(env.finish()[0],-1)
 def test_invented_arguments_fail(self):
  env=ShortRetailEnvironment('113',42,self.cfg('email_lookup'));env.reset();env.step('<tool_call>{"name":"find_user_id_by_email","arguments":{"email":"fake@example.org"}}</tool_call>');self.assertEqual(env.finish()[0],-1)
 def test_partial_progress_is_not_success(self):
  env=ShortRetailEnvironment('113',42,self.cfg('missing_zip'));env.reset();incoming,done,_=env.step('Could you please provide your ZIP code?');self.assertEqual(incoming[0]['role'],'user');self.assertFalse(done);self.assertEqual(env.finish()[0],-1)
 def test_known_fields_are_not_requested_again(self):self.assertFalse(clarification_ok('Please provide your first name, last name, and ZIP code.','zip'))
 def test_email_recovery_requires_correction_request(self):
  self.assertFalse(clarification_ok('Please provide your email again.','email'));self.assertTrue(clarification_ok('Could you please check the correct email address?','email'))
 def test_role_fabrication_is_rejected_even_with_correct_call(self):
  env=ShortRetailEnvironment('113',42,self.cfg('email_lookup'));env.reset();env.step('User: yes\n<tool_call>'+json.dumps(env.expected[0])+'</tool_call>');self.assertEqual(env.finish()[0],-1)
 def test_collector_labels_short_success_separately_from_native_tau2(self):
  self._collector_check(False)
 def test_truncation_fails_immediately_in_short_environment(self):
  self._collector_check(True)
 def _collector_check(self,truncated):
  from tempfile import TemporaryDirectory
  from pathlib import Path
  from tau2_grpo.rl_rollout import collect_episode
  env=ShortRetailEnvironment('113',42,self.cfg('email_lookup'))
  class Controller:
   summaries=[]
   def prepare(self,raw,seed):return {'prompt_ids':[1]}
  class Policy:
   version=0
   def generate(self,prepared,max_tokens,seed):
    return dict(completion_ids=[6],old_logprobs=[-.1],text='<tool_call>'+json.dumps(env.expected[0])+'</tool_call>',ended_with_eos=not truncated)
  with TemporaryDirectory() as temp:
   episode=collect_episode(Policy(),env,Controller(),'113',42,self.cfg('email_lookup'),Path(temp)/'episode.json')
  self.assertTrue(episode['complete'],episode.get('error'));self.assertIsNone(episode['native_reward']);self.assertEqual(episode['reward'],-1 if truncated else 1);self.assertEqual(len(episode['actions']),1)
  self.assertEqual(episode['short_task_success'],not truncated)
if __name__=='__main__':unittest.main()
