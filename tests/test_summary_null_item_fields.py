import unittest,json,copy
from pathlib import Path
from tau2_grpo.rl_context import normalize_provenance
from tau2_grpo.retail_memory import summary_errors,controller_history
class NullItemTests(unittest.TestCase):
 def fixture(self):
  p=Path('outputs/hf-jobs/6afa811929dd05ff0369adb9/bucket/assisted-after-e2e-20261116T040544/smoke/group-0000/episode-11.json');e=json.loads(p.read_text());a=e['summary_attempts'][0];m=json.loads(a['response']['choices'][0]['message']['content']);m,_=normalize_provenance(m,e['messages']);m.pop('confirmation',None);m['observed_history']=controller_history(e['messages']);m['recent_exchange']=dict(source_ids=json.loads(a['request'][1]['content'])['recent_tail_message_ids'],owner='controller');return m,e['messages']
 def test_unknown_null_product_is_not_a_false_claim(self):
  m,raw=self.fixture();self.assertIsNone(next(i for i in m['items'] if i['item_id']=='1421289881')['product_id']);self.assertEqual(summary_errors(m,raw),[])
 def test_wrong_concrete_product_still_rejected(self):
  m,raw=self.fixture();next(i for i in m['items'] if i['item_id']=='1421289881')['product_id']='8310926033';self.assertTrue(any('item_fields_do_not_match' in e for e in summary_errors(m,raw)))
 def test_wrong_price_still_rejected(self):
  m,raw=self.fixture();next(i for i in m['items'] if i['item_id']=='1421289881')['price']=54.85;self.assertTrue(any('item_fields_do_not_match' in e for e in summary_errors(m,raw)))
