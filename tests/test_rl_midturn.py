import json,random,unittest
from pathlib import Path
from dataclasses import replace
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_bank import bank_environment,balanced_order,load_bank,CLARIFICATION_FAMILIES
from tau2_grpo.rl_midturn import MID_PREFIX,derive_case
class MidturnTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=load_settings('configs/rl-retail-midturn-pilot.json');cls.root,cls.rows,_,cls.catalog=load_bank(cls.cfg.scenario_bank)
 def test_all_midturn_reference_suffixes_execute(self):
  for world_no in range(3):
   for family in CLARIFICATION_FAMILIES:
    cid=f'world-{world_no:04d}-{family}';case=self.rows[cid];world=json.loads((self.root/case['world_file']).read_text());derived=derive_case(case,world,self.catalog)
    self.assertEqual(derived['messages'][-1]['role'],'user');self.assertEqual(derived['messages'][-2]['role'],'assistant');self.assertEqual(len(derived['routes'][0]),1)
    env=bank_environment(self.cfg,MID_PREFIX+cid);messages=env.reset();step=derived['routes'][0][0];env.step('<tool_call>'+json.dumps(dict(name=step['name'],arguments=step['arguments']))+'</tool_call>');self.assertEqual(env.finish()[0],1.)
 def test_all_user_facts_and_reply_are_retained(self):
  cid='world-0000-missing_zip';env=bank_environment(self.cfg,MID_PREFIX+cid);m=env.reset();world=json.loads((self.root/self.rows[cid]['world_file']).read_text());u=next(iter(world['users'].values()));text='\n'.join(x['content'] for x in m if x['role']=='user')
  for fact in [u['name']['first_name'],u['name']['last_name'],u['address']['zip']]:self.assertIn(fact,text)
  self.assertNotIn('routes',json.dumps(m));self.assertEqual(env.case['curriculum_start'],'after_customer_reply')
 def test_extra_question_after_reply_is_a_failure(self):
  env=bank_environment(self.cfg,MID_PREFIX+'world-0000-missing_zip');env.reset();env.step('Could you please provide your email address?');self.assertEqual(env.finish()[0],-1.)
 def test_mix_and_splits(self):
  ids=task_ids(self.cfg);self.assertEqual(len(ids),3040);self.assertEqual(sum(x.startswith(MID_PREFIX) for x in ids),640);self.assertFalse(set(ids)&set(task_ids(self.cfg,'validation')))
  order=balanced_order(self.cfg,ids,random.Random(42))[:100];counts={'midturn':0,'opening':0,'retention':0}
  for cid in order:
   category='midturn' if cid.startswith(MID_PREFIX) else 'opening' if self.rows[cid]['family'] in CLARIFICATION_FAMILIES else 'retention';counts[category]+=1
  self.assertEqual(counts,dict(midturn=60,opening=20,retention=20))
 def test_midturn_cannot_be_used_in_original_opening_view(self):
  with self.assertRaises(ValueError):bank_environment(replace(self.cfg,bank_view='opening',bank_sampling='balanced'),MID_PREFIX+'world-0000-missing_zip')
if __name__=='__main__':unittest.main()
