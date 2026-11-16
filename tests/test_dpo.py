import unittest,math,json
from pathlib import Path
import torch
from tau2_grpo.dpo import dpo_loss,read_pairs

class DPOTests(unittest.TestCase):
 def test_initial_policy_equals_reference_gives_log_two(self):
  c=torch.tensor([-3.],requires_grad=True);r=torch.tensor([-7.],requires_grad=True)
  loss,margin=dpo_loss(c,r,c.detach(),r.detach(),.1)
  self.assertAlmostEqual(float(loss.detach()),math.log(2),places=6);self.assertEqual(float(margin.detach()),0.)
 def test_gradient_increases_chosen_and_decreases_rejected(self):
  c=torch.tensor([-3.],requires_grad=True);r=torch.tensor([-7.],requires_grad=True);rc=torch.tensor([-3.],requires_grad=True);rr=torch.tensor([-7.],requires_grad=True)
  loss,_=dpo_loss(c,r,rc,rr,.1);loss.sum().backward();self.assertLess(float(c.grad),0);self.assertGreater(float(r.grad),0);self.assertIsNone(rc.grad);self.assertIsNone(rr.grad)
 def test_pair_data_are_balanced_and_split_by_world(self):
  root=Path('outputs/retail-data/clarification-dpo');a=read_pairs(root/'train.jsonl');b=read_pairs(root/'validation.jsonl')
  self.assertEqual(len(a),300);self.assertEqual(len(b),68);self.assertFalse({r['world_id'] for r in a}&{r['world_id'] for r in b})
  for category in ('missing_ask','supplied_call','retention'):self.assertEqual(sum(r['category']==category for r in a),100)
  self.assertTrue(all(int(r['world_id'].split('-')[1])<80 for r in a))
 def test_supplied_controls_have_the_field_present_in_initial_user_prompt(self):
  rows=read_pairs('outputs/retail-data/clarification-dpo/train.jsonl')
  for r in rows:
   if r['category']!='supplied_call' or r['family']=='failed_email':continue
   self.assertEqual(r['prompt'][-1]['role'],'user')
   self.assertIn('My ',r['prompt'][-1]['content'])
   self.assertIn('<tool_call>',r['chosen']);self.assertNotIn('<tool_call>',r['rejected'])
 def test_eos_prompt_and_padding_do_not_enter_preference_objective(self):
  from transformers import AutoTokenizer
  from tau2_grpo.dpo_data import render_completion
  tok=AutoTokenizer.from_pretrained('outputs/rl-initialization/post-rl-clarification-sft-best19')
  rows=read_pairs('outputs/retail-data/clarification-dpo/validation.jsonl')
  for r in rows[::10]:
   for key,ids in [('chosen','chosen_ids'),('rejected','rejected_ids')]:
    p,c=render_completion(tok,r['prompt'],r[key]);self.assertEqual(p,r['prompt_ids']);self.assertEqual(c,r[ids]);self.assertEqual(c[-1],tok.eos_token_id)
 def test_real_lora_pair_update_keeps_reference_fixed(self):
  from transformers import Cohere2Config,Cohere2ForCausalLM
  from peft import get_peft_model,LoraConfig
  from tau2_grpo.rl_policy import action_logprobs_batch
  torch.manual_seed(42);model=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,sliding_window=32,attention_dropout=0.,eos_token_id=2,bos_token_id=1,pad_token_id=0))
  model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'));model.eval()
  actions=[dict(prompt_ids=[1,4,5],completion_ids=[6,2]),dict(prompt_ids=[1,4,5],completion_ids=[7,8,2])]
  with torch.no_grad():reference=[p.sum().detach() for p in action_logprobs_batch(model,actions,0)]
  frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad};opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.2)
  scores=action_logprobs_batch(model,actions,0);loss,_=dpo_loss(scores[0].sum(),scores[1].sum(),reference[0],reference[1],.1);loss.backward();opt.step()
  with torch.no_grad():scores=action_logprobs_batch(model,actions,0);_,margin=dpo_loss(scores[0].sum(),scores[1].sum(),reference[0],reference[1],.1)
  self.assertGreater(float(margin),0.)
  self.assertTrue(all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen))
if __name__=='__main__':unittest.main()
