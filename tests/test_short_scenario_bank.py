import copy,json,unittest
from pathlib import Path
from tau2_grpo.short_scenario_bank import FAMILIES,make_world,build_case,validate_references,BankEnvironment,fingerprint,matches
class BankTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from tau2.domains.retail.data_model import get_db
  cls.base=get_db();cls.catalog={k:v.model_dump(mode='json') for k,v in cls.base.products.items()};cls.world=make_world(cls.base,0,'59','#W2702727');cls.system=Path('templates/retail_apigen_sft_system.txt').read_text()
 def case(self,family):
  case,_=build_case(self.world,self.catalog,family,self.system,0);return validate_references(case,self.world,self.catalog)
 def test_all30_reference_families_execute(self):
  for family in FAMILIES:
   with self.subTest(family=family):self.assertLessEqual(self.case(family)['max_actor_turns'],4)
 def test_all15_advertised_tools_covered(self):
  names={step['name'] for family in FAMILIES for route in self.case(family)['routes'] for step in route if step['kind']=='call'};self.assertEqual(len(names),15)
 def test_direct_order_lookup_after_authentication_is_a_valid_route(self):
  case=self.case('order_status');self.assertEqual(sorted(len(r) for r in case['routes']),[3,4])
 def test_failed_email_supports_both_corrected_email_and_name_zip(self):
  case=self.case('failed_email');self.assertEqual({r[0]['field'] for r in case['routes']},{'email','zip'})
 def test_no_write_before_confirmation(self):
  case=self.case('cancel_needs_confirmation');env=BankEnvironment(case,self.world,self.catalog);env.reset();before=fingerprint(env.tools.db);target=case['routes'][0][1];env.step('<tool_call>'+json.dumps({'name':target['name'],'arguments':target['arguments']})+'</tool_call>');self.assertEqual(env.finish()[0],-1);self.assertEqual(before,fingerprint(env.tools.db))
 def test_refusal_does_not_cancel_order(self):
  case=self.case('confirmation_declined');env=BankEnvironment(case,self.world,self.catalog);env.reset();before=fingerprint(env.tools.db);env.step('I will leave the order unchanged.');self.assertEqual(env.finish()[0],1);self.assertEqual(before,fingerprint(env.tools.db))
 def test_hidden_zip_and_reply_not_exposed_initially(self):
  case=self.case('missing_zip');zip_code=next(iter(self.world['users'].values()))['address']['zip'];self.assertNotIn(zip_code,json.dumps(case['messages']))
 def test_calculator_accepts_equivalent_expression_but_not_invented_answer_literal(self):
  step=self.case('calculate_total')['routes'][0][0];a,b=step['arguments']['expression'].split(' + ');equivalent={'name':'calculate','arguments':{'expression':f'({b}) + ({a})'}};self.assertTrue(matches(step,'',[equivalent],''));bad={'name':'calculate','arguments':{'expression':str(float(a)+float(b))}};self.assertFalse(matches(step,'',[bad],''))
 def test_status_cannot_claim_two_conflicting_states(self):
  step={'kind':'status','status':'pending','order_id':'#W1'};self.assertTrue(matches(step,'The order is pending, not yet processed.',[],''));self.assertFalse(matches(step,'It is pending but now delivered.',[],''))
 def test_fabricated_roles_not_accepted(self):
  step=self.case('email_lookup')['routes'][0][0];self.assertFalse(matches(step,'User: yes',[{'name':step['name'],'arguments':step['arguments']}],''))
if __name__=='__main__':unittest.main()
