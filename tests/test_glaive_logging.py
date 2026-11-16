import unittest
from tau2_grpo.glaive_logging import rollout_records,COLUMNS

class RolloutLoggingTests(unittest.TestCase):
    def episode(self,i,reward):
        return {'episode_id':str(i),'task_id':'task','scenario':'missing_required','policy_version':0,
                'reward':reward,'complete':True,'messages':[{'role':'user','content':'Hello'}],
                'agent_turns':1,'customer_replied':False,'termination_reason':'end','reward_details':{}}
    def test_every_rollout_has_advantage_and_sample(self):
        episodes=[self.episode(0,0),self.episode(1,1)]
        rows,metrics=rollout_records(episodes,4)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(len(r)==len(COLUMNS) for r in rows))
        self.assertLess(metrics['rollouts/advantage_0'],0)
        self.assertGreater(metrics['rollouts/advantage_1'],0)
        self.assertEqual(episodes[0]['advantage'],metrics['rollouts/advantage_0'])
    def test_tied_group_logs_zero_advantages(self):
        rows,metrics=rollout_records([self.episode(i,1) for i in range(8)],0)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(metrics[f'rollouts/advantage_{i}']==0 for i in range(8)))
