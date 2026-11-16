import unittest
from tau2_grpo.dialogue_roles import assert_speaker
class RoleGateTests(unittest.TestCase):
 def test_rejects_agent_speaking_as_customer(self):
  with self.assertRaisesRegex(ValueError,'customer_perspective'):assert_speaker('assistant',"I don't remember the specific order numbers, but I know I ordered Headphones.")
 def test_rejects_customer_speaking_as_agent(self):
  with self.assertRaisesRegex(ValueError,'agent_perspective'):assert_speaker('user',"No worries. I've located your recent orders. I will process your return.")
 def test_rejects_verbatim_customer_echo(self):
  text='Please help me resolve the issue with this delivery before Friday.'
  with self.assertRaisesRegex(ValueError,'copied_customer'):assert_speaker('assistant',text,[{'role':'user','content':text}])
 def test_accepts_support_actions_and_normal_customer(self):
  assert_speaker('assistant','I will inspect your order and check whether it can be cancelled.')
  assert_speaker('assistant','Could you provide your email address or your name and ZIP code?')
  assert_speaker('user','I ordered headphones and would like to return them.')
  assert_speaker('user','Yes, please proceed with the refund to the original payment method.')
 def test_explicit_quotation_is_not_role_inversion(self):
  assert_speaker('assistant','You said "I ordered headphones"; I will check your order.')
if __name__=='__main__':unittest.main()
