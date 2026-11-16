import unittest
import torch
from tau2_grpo.hint_distillation import teacher_messages,topk_tail_jsd

class HindsightTests(unittest.TestCase):
    def test_feedback_does_not_mutate_history(self):
        history=[{'role':'user','content':'original'}]
        teacher=teacher_messages(history,{'error':'bad'},0,{'diagnosis':'ask'})
        self.assertEqual(history[0]['content'],'original')
        self.assertIn('Tool observation:',teacher[0]['content'])
        self.assertNotIn('Feedback',history[0]['content'])
    def test_equal_distributions_zero(self):
        x=torch.randn(2,7)
        self.assertTrue(torch.allclose(topk_tail_jsd(x,x,3),torch.zeros(2),atol=1e-6))
    def test_teacher_detached_student_gradients(self):
        x=torch.randn(2,7,requires_grad=True);y=torch.randn(2,7,requires_grad=True)
        loss=topk_tail_jsd(x,y,3).mean();loss.backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertIsNone(y.grad)
        self.assertGreaterEqual(float(loss.detach()),0)
        self.assertLessEqual(float(loss.detach()),.694)
    def test_extreme_logits_remain_finite_with_finite_gradients(self):
        x=torch.tensor([[1000.,-1000.,-1000.,0.]],requires_grad=True)
        y=torch.tensor([[-1000.,1000.,-1000.,0.]],requires_grad=True)
        loss=topk_tail_jsd(x,y,3).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNone(y.grad)

    def test_all_support_matches_full_jsd(self):
        x=torch.randn(2,7);y=torch.randn(2,7)
        p=x.softmax(-1);q=y.softmax(-1);m=(p+q)/2
        expected=.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))
        self.assertTrue(torch.allclose(topk_tail_jsd(x,y,7),expected,atol=1e-6))
