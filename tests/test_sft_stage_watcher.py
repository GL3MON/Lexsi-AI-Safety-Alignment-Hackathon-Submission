import json,tempfile,unittest
from pathlib import Path
from scripts.watch_short_to_long_sft import best_adapter

class BestCheckpointTests(unittest.TestCase):
 def test_uses_best_instead_of_latest(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);best=run/'checkpoint-50';latest=run/'checkpoint-100';best.mkdir();latest.mkdir()
   for name in ('adapter_config.json','adapter_model.safetensors'):(best/name).write_text('valid')
   (run/'trainer_state.json').write_text(json.dumps({'best_model_checkpoint':str(best),'best_metric':.2}))
   self.assertEqual(best_adapter(run)[0],best)
 def test_missing_best_never_falls_back(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);(run/'final_adapter').mkdir()
   (run/'trainer_state.json').write_text(json.dumps({'best_model_checkpoint':str(run/'checkpoint-50'),'best_metric':.2}))
   with self.assertRaises(ValueError):best_adapter(run)

if __name__=='__main__':unittest.main()
