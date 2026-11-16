import unittest,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('q',Path('hf_jobs/assisted_after_e2e/queue_watch.py'));q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
class QueueTests(unittest.TestCase):
 def test_wait_for_job_exit(self):self.assertFalse(q.readiness('RUNNING',True,False,{'complete':True})[0])
 def test_do_not_launch_on_failed_parent(self):self.assertFalse(q.readiness('COMPLETED',False,True,{'complete':True})[0])
 def test_do_not_launch_on_infrastructure_failure(self):self.assertFalse(q.readiness('COMPLETED',True,False,{'complete':False})[0])
 def test_scored_failure_is_not_an_infrastructure_failure(self):self.assertTrue(q.readiness('COMPLETED',True,False,{'complete':True,'native_reward':0.})[0])
 def test_ready_after_success(self):self.assertTrue(q.readiness('COMPLETED',True,False,{'complete':True,'native_reward':1.})[0])
