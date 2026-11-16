import unittest
from types import SimpleNamespace
from tau2_grpo.rl_context import FrozenSummarizer
class FeedbackTests(unittest.TestCase):
 def test_retries_receive_actual_validator_feedback(self):
  requests=[]
  def create(**kwargs):
   requests.append(kwargs);return SimpleNamespace(model_dump=lambda:dict(choices=[dict(message=dict(content='{}'))]),choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{}'))])
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)));tok=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1]);s=FrozenSummarizer(client,'frozen',tok);raw=[dict(role='system',content='policy'),dict(role='user',content='request'),dict(role='assistant',content='question'),dict(role='user',content='answer')]
  _,audit=s(raw,2,42,0);audit['validation_errors']=['item_fields_do_not_match_cited_item:1421289881'];s(raw,2,42,1)
  self.assertNotEqual(requests[0]['messages'],requests[1]['messages']);self.assertIn('item_fields_do_not_match',requests[1]['messages'][0]['content']);self.assertIn('null or omitted',requests[1]['messages'][0]['content'])
