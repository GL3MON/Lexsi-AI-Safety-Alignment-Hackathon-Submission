import unittest,random,json
from pathlib import Path
from collections import Counter
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_bank import bank_environment,balanced_order,validation_cases,case_family,CLARIFICATION_FAMILIES
from tau2_grpo.medium_scenario_bank import reference_text,MediumEnvironment
class IntegrationTests(unittest.TestCase):
 def setUp(self):self.cfg=load_settings('configs/rl-retail-medium.json')
 def test_split_and_mix(self):
  ids=task_ids(self.cfg);validation=task_ids(self.cfg,'validation');test=task_ids(self.cfg,'test');self.assertEqual(len(ids),3040);self.assertFalse(set(ids)&set(validation));self.assertFalse(set(test)&set(validation));self.assertFalse(set(ids)&set(test));order=balanced_order(self.cfg,ids,random.Random(244))[:50];counts=Counter('medium' if '-medium-' in c else 'clarification' if case_family(self.cfg,c) in CLARIFICATION_FAMILIES else 'retention' for c in order);self.assertEqual(dict(counts),dict(medium=30,clarification=10,retention=10));self.assertEqual(len(set(order)),50)
 def test_validation114(self):
  ids=validation_cases(self.cfg);self.assertEqual(len(ids),114);self.assertEqual(sum('-medium-' in i for i in ids),24);self.assertFalse(set(ids)&set(task_ids(self.cfg)))
 def test_real_routes_via_rl_loader(self):
  from tau2_grpo.medium_scenario_bank import FAMILIES
  for family in FAMILIES:
   cid='world-0000-medium-'+family;env=bank_environment(self.cfg,cid);self.assertIsInstance(env,MediumEnvironment);initial=env.reset();self.assertEqual([m['role'] for m in initial],['system','user'])
   for s in env.case['routes'][0]:env.step(reference_text(s))
   self.assertTrue(env.success,cid);self.assertEqual(env.finish()[0],1)
 def test_originals_remain_available(self):
  env=bank_environment(self.cfg,'world-0000-missing_zip');self.assertNotIsInstance(env,MediumEnvironment);self.assertEqual(case_family(self.cfg,'world-0000-medium-cancel_email'),'medium_cancel_email')
