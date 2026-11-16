import unittest,json
from pathlib import Path
from tau2_grpo.rl_bank import load_bank
from tau2_grpo.medium_scenario_bank import MediumEnvironment,build_medium_case,reference_text,FAMILIES
class MediumTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root,rows,pools,cls.catalog=load_bank('outputs/retail-data/short-scenario-bank-3000');source=rows[pools['train'][0]];cls.world=json.loads((root/source['world_file']).read_text());cls.system=Path('templates/retail_apigen_sft_system.txt').read_text()
 def make(self,family='status_email'):
  case=build_medium_case(self.world,self.catalog,self.system,family);return case,MediumEnvironment(case,self.world,self.catalog)
 def test_entire_routes_generated(self):
  for family in FAMILIES:
   case,env=self.make(family);self.assertEqual([m['role'] for m in case['messages']],['system','user']);self.assertTrue(4<=case['reference_actor_turns']<=8);env.reset()
   for step in case['routes'][0]:env.step(reference_text(step))
   self.assertTrue(env.success,family)
 def test_invalid_call_can_be_corrected(self):
  case,env=self.make();env.reset();_,done,_=env.step('<tool_call>{"name":"invented_lookup","arguments":{}}</tool_call>');self.assertFalse(done)
  for step in case['routes'][0]:env.step(reference_text(step))
  self.assertTrue(env.success);self.assertEqual(env.repairs,1)
 def test_invalid_argument_is_not_executed(self):
  case,env=self.make();env.reset();_,done,_=env.step('<tool_call>{"name":"find_user_id_by_email","arguments":{"address":"unknown"}}</tool_call>');self.assertFalse(done);self.assertEqual(env.routes,case['routes']);self.assertEqual(env.repairs,1)
 def test_repeated_failure_bounded(self):
  _,env=self.make();env.reset();bad='<tool_call>{"name":"invented_lookup","arguments":{}}</tool_call>';env.step(bad);_,done,_=env.step(bad);self.assertTrue(done);self.assertFalse(env.success)
 def test_write_before_confirmation_fails(self):
  case,env=self.make('cancel_email');env.reset()
  for step in case['routes'][0][:3]:env.step(reference_text(step))
  env.step(reference_text(case['routes'][0][-1]));self.assertTrue(env.done);self.assertFalse(env.success);self.assertEqual(next(iter(env.tools.db.orders.values())).status,'pending')

 def test_malformed_json_can_be_repaired(self):
  case,env=self.make();env.reset();_,done,_=env.step('<tool_call>{"name":"find_user_id_by_email","arguments":{</tool_call>');self.assertFalse(done)
  for step in case['routes'][0]:env.step(reference_text(step))
  self.assertTrue(env.success)

 def test_verified_progress_is_not_full_success(self):
  case,env=self.make();env.reset();env.step(reference_text(case['routes'][0][0]));reward,info=env.finish();self.assertAlmostEqual(reward,-.5);self.assertFalse(info['short_success']);self.assertEqual(env.reward_details['verified_steps'],1)
