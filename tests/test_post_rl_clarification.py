import json,random,unittest
from dataclasses import replace
from pathlib import Path
from tau2_grpo.rl_bank import load_bank,balanced_order,CLARIFICATION_FAMILIES
from tau2_grpo.rl_config import load_settings
from scripts.curate_post_rl_clarification_sft import QUESTIONS,render_action
from tau2_grpo.short_scenario_bank import matches

class ClarificationCurriculumTests(unittest.TestCase):
 def test_varied_questions_only_ask_the_target_field(self):
  for field in ('email','zip','first_name','last_name'):
   for i in range(len(QUESTIONS)):
    step=dict(kind='ask',field=field,correction=field=='email')
    text=render_action(step,i)
    self.assertTrue(matches(step,text,[],''),(field,text))
 def test_continued_rl_mixes_two_clarification_groups_per_retention(self):
  cfg=replace(load_settings('hf_jobs/short_bank_rl/config.json'),bank_sampling='clarification_mix')
  _,rows,pools,_=load_bank(cfg.scenario_bank);order=balanced_order(cfg,pools['train'],random.Random(42))
  families=[rows[c]['family'] for c in order[:90]]
  self.assertEqual(sum(f in CLARIFICATION_FAMILIES for f in families),60)
  self.assertEqual(len(set(families)),30)
  self.assertTrue(set(order)<=set(pools['train']))
 def test_completed_curated_dataset_and_masks(self):
  root=Path('outputs/retail-data/post-rl-clarification200-retention100');m=json.loads((root/'manifest.json').read_text());self.assertTrue(m['complete']);self.assertLessEqual(m['max_tokens'],8192)
  rows=[json.loads(l) for l in (root/'train.jsonl').read_text().splitlines()]
  self.assertEqual(sum(r['category']=='clarification' for r in rows),200);self.assertEqual(sum(r['category']=='retention' for r in rows),100)
  for r in rows:
   self.assertTrue(r['verification']['executed']);self.assertTrue(all(i>=r['prefix_message_count'] for i in r['loss_message_indices']))
   self.assertTrue(all(r['messages'][i]['role']=='assistant' for i in r['loss_message_indices']))
   self.assertLess(int(r['world_id'].split('-')[1]),80)
if __name__=='__main__':unittest.main()
