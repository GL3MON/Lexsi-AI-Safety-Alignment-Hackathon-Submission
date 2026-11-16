import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from tau2_grpo.sft_backends import SFTBackend
from tau2_grpo.sft_data import tokenize_assistant_only

class ByteTokenizer:
 def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict):
  return list((''.join('<'+m['role']+'>'+m['content']+'<end>' for m in messages)+('<assistant>' if add_generation_prompt else '')).encode())

class BackendTests(unittest.TestCase):
 def exercise(self,name,root):
  tokenizer=ByteTokenizer();records=[{'id':'overlap','loss_message_indices':[4],'messages':[{'role':'system','content':'policy'},{'role':'user','content':'request'},{'role':'assistant','content':'earlier context'},{'role':'user','content':'<tool_response>{"ok":true}</tool_response>'},{'role':'assistant','content':'target'}]}]
  encoded=[tokenize_assistant_only(tokenizer,records[0]['messages'],1000,[4])[0]];calls=[]
  class Trainer:
   def train(self,**kwargs):calls.append(kwargs);return SimpleNamespace(global_step=2,training_loss=1.,metrics={})
  def build(rows):
   self.assertEqual(rows,encoded);self.assertTrue(any(x==-100 for x in rows[0]['labels']));return Trainer()
  driver=SFTBackend(name,records=records,encoded=encoded,tokenizer=tokenizer,max_length=1000,run_dir=root)
  result=driver.train(build_trainer=build,on_ready=lambda t:None,resume='checkpoint-1')
  self.assertEqual(result.global_step,2);self.assertEqual(calls,[{'resume_from_checkpoint':'checkpoint-1'}]);return driver
 def test_trl_keeps_precomputed_labels(self):
  with tempfile.TemporaryDirectory() as d:self.assertIsNone(self.exercise('trl',Path(d)).owner)
 def test_native_aligntune_lifecycle_and_mask(self):
  from aligntune.core.trainer_base import UnifiedTrainerBase
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);driver=self.exercise('aligntune',root)
   self.assertIsInstance(driver.owner,UnifiedTrainerBase)
   self.assertEqual([e['event'] for e in json.loads((root/'aligntune_events.json').read_text())],['init','train_begin','train_end'])
   self.assertTrue((root/'lexsi_provenance.json').exists())
   self.assertTrue(json.loads((root/'backend_audit.json').read_text())['success'])
 def test_old_backend_argument_resolves_to_new_framework(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(self.exercise('agenttune',Path(d)).name,'aligntune')
