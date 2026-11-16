import json,unittest
from pathlib import Path
from tau2_grpo.tau2_curriculum import make_curriculum,curriculum_order
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_rewards import ToolRewardTracker
class CurriculumTests(unittest.TestCase):
 def test_only_rl_ids_and_stages(self):
  c=load_settings('configs/rl-retail-tau2-curriculum.json');data=json.loads(Path(c.curriculum_file).read_text());order=curriculum_order(c,task_ids(c));self.assertEqual(len(order),30);self.assertTrue(set(order)<=set(task_ids(c)));self.assertFalse(set(order)&set(task_ids(c,'validation')));self.assertEqual([x['stage'] for x in data['schedule'][:10]],['easy']*10);self.assertEqual([x['stage'] for x in data['schedule'][20:]],['hard']*10);self.assertEqual(sum(x['retention'] for x in data['schedule']),4);self.assertTrue(all(x['reference_actions']<=8 for x in data['ranked_tasks']));self.assertEqual({x['task_id'] for x in data['deferred_tasks']},{'30','54'})
 def test_confirmation_required_for_mutation_credit(self):
  from tau2_grpo.rl_rewards import has_confirmation
  self.assertFalse(has_confirmation([dict(role='user',content='Cancel my order')]))
  self.assertTrue(has_confirmation([dict(role='assistant',content='Please confirm cancellation of the whole order'),dict(role='user',content='Yes, go ahead')]))
  self.assertFalse(has_confirmation([dict(role='assistant',content='Please confirm cancellation'),dict(role='user',content='No, do not cancel')]))
