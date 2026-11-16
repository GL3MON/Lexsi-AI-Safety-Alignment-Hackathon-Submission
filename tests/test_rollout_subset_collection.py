import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_transformers_workers import collect_group
class Connection:
 def recv(self):return self.message
 def close(self):pass
class Context:
 def Pipe(self):
  parent=Connection();child=SimpleNamespace(parent=parent,close=lambda:None);return parent,child
 def Process(self,target,args):
  child,raw,version,tid,seed,file=args
  def start():Path(file).parent.mkdir(parents=True,exist_ok=True);Path(file).write_text(json.dumps(dict(task_id=tid,seed=seed,complete=True)));child.parent.message=('done',file)
  return SimpleNamespace(start=start,join=lambda **kw:None,is_alive=lambda:False)
class SubsetTests(unittest.TestCase):
 def test_nonzero_index_preserves_seed_and_filename(self):
  with tempfile.TemporaryDirectory() as d,patch('tau2_grpo.rl_transformers_workers.multiprocessing.get_context',return_value=Context()),patch('tau2_grpo.rl_transformers_workers.wait',side_effect=lambda connections,timeout:list(connections)):
   rows=collect_group(SimpleNamespace(version=0),RetailRLConfig(group_size=16,rollout_workers=16),'13',345,d,rollout_indices=[11]);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['seed'],356);self.assertTrue((Path(d)/'episode-11.json').exists())
 def test_subset_cannot_be_used_for_training_reference(self):
  with self.assertRaises(ValueError):collect_group(SimpleNamespace(version=0),RetailRLConfig(),'13',345,'/unused',with_reference=True,rollout_indices=[1])
