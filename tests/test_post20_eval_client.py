import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from tau2_grpo.rl_post20_eval import LocalFrozenClient
class Inputs(dict):
 def to(self,device):return self
class Output:
 def __init__(self,tokens):self.tokens=tokens
 def __getitem__(self,index):return self.tokens
class ClientTests(unittest.TestCase):
 def client(self,tokens):
  c=LocalFrozenClient.__new__(LocalFrozenClient);c.torch=SimpleNamespace(manual_seed=lambda seed:None,inference_mode=nullcontext)
  c.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:Inputs(input_ids=SimpleNamespace(shape=(1,5))),eos_token_id=6,decode=lambda *a,**kw:'Customer response')
  def generate(**kw):self.kw=kw;return Output(tokens)
  c.model=SimpleNamespace(generate=generate,generation_config=SimpleNamespace(eos_token_id=[6]));return c
 def test_exact_response_interface_and_greedy(self):
  c=self.client([42,6]);r=c.create(model='frozen',messages=[dict(role='user',content='Hello')],max_tokens=20,seed=5)
  self.assertEqual(r.choices[0].finish_reason,'stop');self.assertEqual(r.model_dump()['choices'][0]['message']['content'],'Customer response');self.assertFalse(self.kw['do_sample']);self.assertEqual(self.kw['max_new_tokens'],20)
 def test_truncation_exposed(self):
  r=self.client([42,43]).create(model='frozen',messages=[],max_tokens=2);self.assertEqual(r.choices[0].finish_reason,'length')

 def test_json_wrapper_is_removed_only_for_valid_entire_json(self):
  from tau2_grpo.rl_post20_eval import json_content
  self.assertEqual(json_content('```json\n{"ok": true}\n```'),'{"ok": true}')
  bad='```json\n{broken}\n```';self.assertEqual(json_content(bad),bad)
  extra='Here you go: {"ok":true}';self.assertEqual(json_content(extra),extra)
