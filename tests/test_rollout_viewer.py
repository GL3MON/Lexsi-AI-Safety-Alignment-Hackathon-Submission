import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('rollout_viewer',Path('scripts/serve_rollouts.py'));viewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(viewer)

class ViewerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=viewer.ROOT;viewer.ROOT=self.root
  self.job=dict(job_id='test-run',task='short-bank',groups=60)
  self.store=viewer.Store([self.job]);self.bucket=self.store.folder('test-run')/'bucket'
  self.group=self.bucket/'run/group-0000';self.group.mkdir(parents=True)
  (self.group.parent/'groups.json').write_text(json.dumps([dict(group=0,task_id='world-email',rewards=[-1]*8,skipped_tied_group=True,optimizer_steps_total=0)]))
 def tearDown(self):viewer.ROOT=self.old;self.temp.cleanup()
 def test_tied_advantages_display_as_zero_and_preserve_negative_rewards(self):
  (self.group/'episode-00.txt').write_text('USER:\nHi\n\nASSISTANT:\nHello')
  state=self.store.index('test-run');self.assertEqual(state['episodes'][0]['reward'],-1);self.assertEqual(state['episodes'][0]['advantage'],0.)
  self.assertEqual(state['stats']['tied_groups'],1);self.assertEqual(state['stats']['failures'],8)
 def test_json_messages_keep_roles_and_strip_large_token_arrays(self):
  raw=dict(messages=[dict(role='tool',content='result')],actions=[dict(prompt_ids=[1,2],completion_ids=[3],old_logprobs=[-.5],raw_message_count=0)],complete=True,reward=1)
  (self.group/'episode-00.json').write_text(json.dumps(raw));e=self.store.episode('test-run','run/group-0000/episode-00')
  self.assertEqual(e['messages'][0]['role'],'tool');self.assertEqual(e['actions'][0]['prompt_tokens'],2);self.assertNotIn('prompt_ids',e['actions'][0])
  self.assertIn('prompt_ids',self.store.episode('test-run','run/group-0000/episode-00',raw=True)['actions'][0])
 def test_traversal_and_unknown_job_rejected(self):
  with self.assertRaises(ValueError):self.store.episode('test-run','../../../../secret')
  with self.assertRaises(ValueError):self.store.index('unknown')
 def test_text_fallback_keeps_assistant_and_tool_boundaries(self):
  (self.group/'episode-00.txt').write_text('USER:\nHi\n\nASSISTANT:\n<tool_call>{}</tool_call>\n\nTOOL:\n{}')
  e=self.store.episode('test-run','run/group-0000/episode-00');self.assertEqual([m['role'] for m in e['messages']],['user','assistant','tool'])
 def test_partial_json_uses_text_without_breaking_index(self):
  (self.group/'episode-00.json').write_text('{');(self.group/'episode-00.txt').write_text('USER:\nHello')
  self.assertEqual(len(self.store.index('test-run')['episodes']),1);self.assertTrue(self.store.episode('test-run','run/group-0000/episode-00')['text_only'])
if __name__=='__main__':unittest.main()
