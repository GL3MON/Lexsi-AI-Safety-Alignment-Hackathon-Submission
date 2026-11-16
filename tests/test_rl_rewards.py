import json,unittest
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_rewards import component_sum,ToolRewardTracker,shaped_reward

SCHEMA={'find_user_id_by_email':{'type':'object','properties':{'email':{'type':'string'}},'required':['email']},'get_user_details':{'type':'object','properties':{'user_id':{'type':'string'}},'required':['user_id']}}
def call(name,**args):return '<tool_call>'+json.dumps({'name':name,'arguments':args})+'</tool_call>'
def result(name,value,error=False):return {'role':'tool','name':name,'error':error,'content':json.dumps({'name':name,'content':value})}
class RewardsTests(unittest.TestCase):
 def test_sum_partial_checks_without_empty_category_credit(self):
  info={'db_check':{'db_reward':0},'action_checks':[{'action':{'action_id':'a'},'action_reward':1},{'action':{'action_id':'b'},'action_reward':0}],'nl_assertions':[],'reward_breakdown':{'DB':0,'NL_ASSERTION':1}}
  total,rows=component_sum(info);self.assertEqual(total,1);self.assertEqual(len(rows),3)
 def test_native_and_component_rewards_not_double_counted(self):
  total,rows=component_sum({'reward':1,'db_check':{'db_reward':1},'reward_breakdown':{'DB':1},'nl_assertions':[{'nl_assertion':'told user','met':True}]});self.assertEqual(total,2)
 def test_no_provenance_from_schema_examples_or_agent_claims(self):
  tracker=ToolRewardTracker(RetailRLConfig())
  prior=[{'role':'system','content':'user@example.com'},{'role':'assistant','content':'Use user@example.com'},{'role':'user','content':'Please cancel my order.'}]
  tracker.record(call('find_user_id_by_email',email='user@example.com'),prior,[result('find_user_id_by_email','somebody')],SCHEMA)
  self.assertFalse(tracker.events[0]['arguments_grounded']);self.assertIsNone(tracker.authenticated_id);self.assertEqual(tracker.score()['argument_bonus'],0)
 def test_correct_copied_email_then_returned_id(self):
  tracker=ToolRewardTracker(RetailRLConfig());prior=[{'role':'user','content':'My email is first.last+orders@example.org.'}]
  response=result('find_user_id_by_email','user_verified_0007')
  tracker.record(call('find_user_id_by_email',email='first.last+orders@example.org'),prior,[response],SCHEMA)
  tracker.record(call('get_user_details',user_id='user_verified_0007'),prior+[response],[result('get_user_details',{'user_id':'user_verified_0007'})],SCHEMA)
  self.assertEqual(tracker.score()['argument_bonus'],.5);self.assertEqual(tracker.score()['max_failure_streak'],0)
 def test_failed_call_streak_penalty_and_success_reset(self):
  tracker=ToolRewardTracker(RetailRLConfig());prior=[{'role':'user','content':'Use wrong@example.org or correct@example.org.'}]
  for _ in range(3):tracker.record(call('find_user_id_by_email',email='wrong@example.org'),prior,[result('find_user_id_by_email','Error: User not found',True)],SCHEMA)
  self.assertEqual([e['failure_penalty'] for e in tracker.events],[0,.25,.5])
  tracker.record(call('find_user_id_by_email',email='correct@example.org'),prior,[result('find_user_id_by_email','real_id')],SCHEMA)
  self.assertEqual(tracker.streak,0)
 def test_error_echo_does_not_launder_invented_argument(self):
  tracker=ToolRewardTracker(RetailRLConfig());prior=[{'role':'user','content':'I do not remember my email.'},result('find_user_id_by_email','Error: ghost@example.org was not found',True)]
  tracker.record(call('find_user_id_by_email',email='ghost@example.org'),prior,[result('find_user_id_by_email','Error: User not found',True)],SCHEMA)
  self.assertFalse(tracker.events[0]['arguments_grounded'])
 def test_duplicate_success_has_no_additional_bonus(self):
  tracker=ToolRewardTracker(RetailRLConfig());prior=[{'role':'user','content':'My email is x@example.org.'}]
  for _ in range(3):tracker.record(call('find_user_id_by_email',email='x@example.org'),prior,[result('find_user_id_by_email','x_id')],SCHEMA)
  self.assertEqual(tracker.score()['unique_correct_calls'],1);self.assertAlmostEqual(tracker.score()['argument_bonus'],.5/3)
 def test_schema_error_and_malformed_call_are_negative(self):
  tracker=ToolRewardTracker(RetailRLConfig())
  for text in ('<tool_call>{bad}</tool_call>',call('made_up_tool',x='y')):tracker.record(text,[],[result('__runtime__',{'error':'invalid'})],SCHEMA)
  self.assertEqual(tracker.score()['max_failure_streak'],2);self.assertGreater(tracker.score()['incorrect_argument_penalty'],0)
 def test_positive_shaping_is_not_mistaken_for_task_success(self):
  import tempfile
  from pathlib import Path
  from tau2_grpo.rl_reporting import write_pass_at_k_report
  e={'task_id':'x','reward':1.,'native_reward':0.,'complete':True,'episode_id':'one','seed':42,'messages':[],'actions':[],'summaries':[]}
  with tempfile.TemporaryDirectory() as d:
   report=write_pass_at_k_report(Path(d),[e],1)
  self.assertFalse(report['observed_pass_at_k']);self.assertEqual(report['samples'][0]['training_reward'],1.)
 def test_native_config_remains_default_for_benchmark(self):self.assertEqual(RetailRLConfig().reward_mode,'native')

if __name__=='__main__':unittest.main()
