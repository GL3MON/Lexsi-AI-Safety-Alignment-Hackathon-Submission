import unittest,json
from pathlib import Path
from collections import defaultdict,Counter
from tau2_grpo.sft_config import load_settings
class DiagnosticTests(unittest.TestCase):
 def test_only_one_supplied_fact_changes_matched_prompts(self):
  root=Path('outputs/retail-data/clarification-diagnostic20');worlds=[]
  for split in ['train','heldout']:
   rows=[json.loads(l) for l in (root/(split+'.jsonl')).read_text().splitlines()];self.assertEqual(len(rows),40);self.assertEqual(Counter(r['category'] for r in rows),{'ask':20,'call':20});groups=defaultdict(dict);worlds.append({r['world_id'] for r in rows})
   for r in rows:groups[r['pair_id']][r['category']]=r;self.assertEqual(r['loss_message_indices'],[2]);self.assertEqual(len(r['messages']),3)
   for g in groups.values():
    absent=g['ask']['messages'][1]['content'];supplied=g['call']['messages'][1]['content'];self.assertTrue(supplied.startswith(absent+' My '));self.assertEqual(g['ask']['messages'][0],g['call']['messages'][0])
  self.assertFalse(worlds[0]&worlds[1])
 def test_final_step_used_and_heldout_never_replaced_by_train(self):
  cfg=load_settings('configs/sft-clarification-diagnostic20.json');self.assertFalse(cfg.overfit_debug);self.assertFalse(cfg.load_best_model_at_end);self.assertEqual(cfg.max_steps,50);self.assertTrue(cfg.example_normalized_loss)
if __name__=='__main__':unittest.main()

class DiagnosticContextTests(unittest.TestCase):
 def test_independent_cases_render_their_own_prompt_and_not_the_first_case(self):
  from transformers import AutoTokenizer
  from tau2_grpo.rl_config import RetailRLConfig
  from tau2_grpo.sft_diagnostic import prepare_decision
  tok=AutoTokenizer.from_pretrained('outputs/rl-initialization/post-rl-clarification-sft-best19');cfg=RetailRLConfig(response_tokens=192,prompt_budget=7680)
  rows=[json.loads(l) for l in Path('outputs/retail-data/clarification-diagnostic20/train.jsonl').read_text().splitlines()][:4]
  prepared=[prepare_decision(tok,r['messages'][:2],cfg) for r in rows]
  self.assertEqual(len({tuple(p['prompt_ids']) for p in prepared}),len(rows))
  for r,p in zip(rows,prepared):self.assertEqual(p['messages'][1]['content'],r['messages'][1]['content']);self.assertIn(r['messages'][1]['content'],p['rendered_prompt'])
