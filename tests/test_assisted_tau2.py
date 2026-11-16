import json,unittest,tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from tau2_grpo.assisted_tau2 import select_assistance,prime_tracker
from tau2_grpo.rl_config import load_settings
from tau2_grpo.rl_rewards import ToolRewardTracker,shaped_reward
class AssistanceTests(unittest.TestCase):
 def test_advancement_requires_observed_success(self):
  self.assertEqual(select_assistance([],0)['assistance_level'],'finish')
  fail=[dict(assistance_level='finish',assistance_primary_level='finish',native_rewards=[0]*16)]*6;self.assertEqual(select_assistance(fail,6)['assistance_level'],'finish')
  good=[dict(assistance_level='finish',assistance_primary_level='finish',native_rewards=[1]*8+[0]*8)]*4;self.assertEqual(select_assistance(good,4)['assistance_primary_level'],'find');self.assertEqual(select_assistance(good,4)['assistance_level'],'finish');self.assertTrue(select_assistance(good,4)['assistance_retention']);self.assertEqual(select_assistance(good,5)['assistance_level'],'find')
 def test_provided_reference_action_does_not_earn_policy_credit(self):
  cfg=load_settings('configs/rl-retail-tau2-assisted.json');t=ToolRewardTracker(cfg);r,d=shaped_reward(0,{'reward_info':{'action_checks':[{'action':{'name':'get_user_details','arguments':{'user_id':'u'},'action_id':'provided'},'action_reward':1}]}},t);self.assertEqual(r,0);self.assertEqual(d['excluded_unsampled_action_credits'],['provided'])
 def test_real_tau2_readonly_prefix_and_suffix_mask(self):
  from transformers import AutoTokenizer
  from tau2_grpo.rl_tau2 import RetailEnvironment
  from tau2_grpo.rl_context import ContextController
  from tau2_grpo.rl_rollout import collect_episode
  cfg=load_settings('configs/rl-retail-tau2-assisted.json');tok=AutoTokenizer.from_pretrained(cfg.initial_adapter);tok.chat_template=Path(cfg.chat_template_file).read_text()
  class External:
   def __init__(self):self.cfg=cfg;self.audit=[];self.n=0
   def request(self,messages,max_tokens,purpose,seed,json_mode=False):
    self.n+=1;last=messages[-1]['content'].lower()
    if self.n==1:text='I want to exchange my Fleece Jacket for a large red one with a half zipper.'
    elif 'email' in last and 'provide' in last:text='My email is yusuf.hernandez8836@example.com.'
    elif 'confirm' in last:text='Yes, I confirm those exact changes and payment method.'
    else:text='Thank you. ###STOP###'
    self.audit.append(dict(purpose=purpose,text=text));return text
  env=RetailEnvironment('85',42,cfg,External())
  target=dict(name='modify_pending_order_items',arguments=dict(order_id='#W2466703',item_ids=['9385662952'],new_item_ids=['8733974883'],payment_method_id='paypal_7529813'))
  class Policy:
   tokenizer=tok;version=0;index=0
   def generate(self,prepared,max_tokens,seed):
    texts=['Please confirm: change the Fleece Jacket to the large red half-zip variant using your saved PayPal method?','<tool_call>'+json.dumps(target)+'</tool_call>','Your jacket modification is complete.'];text=texts[self.index];self.index+=1;tokens=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id];return dict(text=text,completion_ids=tokens,old_logprobs=[-1.]*len(tokens),ended_with_eos=True)
  def no_summary(*a,**kw):raise RuntimeError('Unexpected reference context overflow')
  controller=ContextController(tok,no_summary,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
  with tempfile.TemporaryDirectory() as d:
   e=collect_episode(Policy(),env,controller,'85',42,cfg,Path(d)/'episode.json');self.assertTrue(e['complete'],e.get('traceback'));self.assertEqual(e['native_reward'],1.);self.assertEqual(len(e['actions']),3);self.assertGreater(e['actions'][0]['raw_message_count'],2);self.assertTrue(e['assistance']['database_unchanged']);self.assertFalse(e['assistance']['policy_targets']);self.assertTrue(all(m.get('tool_calls') for m in e['messages'][:e['actions'][0]['raw_message_count']] if m['role']=='assistant' and '<tool_call>' in m.get('content','')));self.assertEqual(len(e['tool_reward_audit']),1);self.assertTrue(e['tool_reward_audit'][0]['arguments_grounded']);self.assertEqual(e['reward_details']['argument_bonus'],0.)
