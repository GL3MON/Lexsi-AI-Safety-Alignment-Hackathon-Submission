import unittest
from types import SimpleNamespace
from tau2_grpo.sft_data import tool_call_count,order_by_tool_calls

def row(n):
 return {'id':str(n),'messages':[{'role':'system','content':'<tool_call>{"name":"fake","arguments":{}}</tool_call>'}]+[{'role':'assistant','content':'I will check.\n<tool_call>{"name":"lookup","arguments":{}}</tool_call>'} for _ in range(n)]+[{'role':'user','content':'<tool_response>{}</tool_response>'}]}

class CurriculumTests(unittest.TestCase):
 def test_actual_assistant_calls_only(self):self.assertEqual(tool_call_count(row(3)),3)
 def test_increasing_counts_without_changing_members(self):
  rows=[row(5),row(4),row(6),row(4)]
  ordered=order_by_tool_calls(rows)
  self.assertEqual([tool_call_count(r) for r in ordered],[4,4,5,6])
  self.assertCountEqual(map(id,rows),map(id,ordered))
 def test_transformers_sequential_sampler_preserves_order(self):
  from transformers import Trainer
  rows=[row(4),row(4),row(5)]
  trainer=SimpleNamespace(args=SimpleNamespace(train_sampling_strategy='sequential'),train_dataset=rows)
  sampler=Trainer._get_train_sampler(trainer,rows)
  self.assertEqual(list(sampler),[0,1,2])

if __name__=='__main__':unittest.main()
