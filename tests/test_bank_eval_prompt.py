import copy,unittest
from tau2_grpo.rl_bank_eval import verify_episode_prompt
class EvalPromptTests(unittest.TestCase):
 def episode(self,text,ids):
  m=[dict(role='system',content='policy'),dict(role='user',content=text)]
  return dict(messages=m+[dict(role='assistant',content='reply')],actions=[dict(messages=copy.deepcopy(m),prompt_ids=ids,raw_message_count=2)])
 def test_distinct_cases_have_distinct_prompts(self):
  seen=set();verify_episode_prompt(self.episode('case A',[1,2]),seen);verify_episode_prompt(self.episode('case B',[1,3]),seen);self.assertEqual(len(seen),2)
 def test_reusing_first_prompt_is_rejected(self):
  seen=set();verify_episode_prompt(self.episode('case A',[1,2]),seen)
  with self.assertRaises(ValueError):verify_episode_prompt(self.episode('case B',[1,2]),seen)
 def test_wrong_rendered_customer_rejected(self):
  e=self.episode('case A',[1,2]);e['actions'][0]['messages'][1]['content']='case B'
  with self.assertRaises(ValueError):verify_episode_prompt(e,set())
if __name__=='__main__':unittest.main()
