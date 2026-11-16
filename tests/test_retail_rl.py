import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tau2_grpo.rl_config import RetailRLConfig,task_ids
from tau2_grpo.rl_context import ContextController,student_messages
from tau2_grpo.rl_rollout import collect_episode
from tau2_grpo.rl_objective import group_advantages,update_group


class CharacterTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,**kwargs):
        text=''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages)
        if add_generation_prompt:text+='<assistant>'
        return list(text.encode()) if tokenize else text
    def encode(self,text,**kwargs):return list(text.encode())


class HarnessTests(unittest.TestCase):
    def test_multiple_compactions_keep_one_environment_reward_and_all_actions(self):
        cfg=replace(RetailRLConfig(),context_limit=1000,prompt_budget=900,response_tokens=100,max_actions=14)
        seen=[]
        def summary(raw,tail_start,seed,attempt,prior=None):
            seen.append(copy.deepcopy(raw))
            self.assertEqual(raw[-1]['role'],'user')
            return {},dict(request_history_count=len(raw))
        controller=ContextController(CharacterTokenizer(),summary,1000,100,900)
        class Env:
            resets=0;scores=0;steps=0;closed=False
            def reset(self):
                self.resets+=1
                return [{'role':'system','content':'policy'},{'role':'user','content':'opening request'}]
            def step(self,text):
                self.steps+=1
                return [{'role':'user','content':'feedback '+str(self.steps)+' '+'u'*70}],self.steps==14,None
            def finish(self):self.scores+=1;return 1.0,{'reward_info':{'reward':1}}
            def close(self):self.closed=True
        class Policy:
            version=3
            def generate(self,prepared,max_tokens,seed):
                self.asserted=True
                return dict(completion_ids=[2,3],old_logprobs=[-1.,-2.],text='answer '+'a'*70,ended_with_eos=True)
        env=Env()
        with tempfile.TemporaryDirectory() as temp:
            record=collect_episode(Policy(),env,controller,'59',42,cfg,Path(temp)/'episode.json')
        self.assertTrue(record['complete'],record.get('error'))
        self.assertEqual((env.resets,env.scores,env.steps),(1,1,14))
        self.assertTrue(env.closed)
        self.assertGreaterEqual(len(record['summaries']),3)
        self.assertEqual(len(record['actions']),14)
        for action in record['actions']:
            self.assertLessEqual(len(action['prompt_ids'])+100,1000)
            self.assertEqual(action['prompt_ids'],controller.tokenizer.encode(action['rendered_prompt']))
            self.assertEqual(action['messages'][0],record['messages'][0])
        # No later action or reward was included in a summary input.
        for history in seen:
            self.assertEqual(history,record['messages'][:len(history)])

    def test_infrastructure_failure_has_null_reward_and_is_not_trainable(self):
        class Env:
            def reset(self):raise RuntimeError('simulator failed')
            def close(self):pass
        with tempfile.TemporaryDirectory() as temp:
            result=collect_episode(SimpleNamespace(version=0),Env(),SimpleNamespace(summaries=[]),'59',1,RetailRLConfig(),Path(temp)/'e.json')
        self.assertIsNone(result['reward']);self.assertFalse(result['complete'])
        with self.assertRaises(ValueError):group_advantages([result,dict(result,episode_id='other')])

    def test_invalid_model_call_is_preserved_for_runtime_feedback(self):
        raw=[{'role':'assistant','content':'<tool_call>{bad}</tool_call>'},
             {'role':'tool','name':'__runtime__','content':'{"name":"__runtime__","content":{"error":"bad JSON"}}'}]
        view=student_messages(raw)
        self.assertEqual(view[0]['content'],raw[0]['content'])
        self.assertEqual(view[1]['role'],'user');self.assertIn('<tool_response>',view[1]['content'])

    def test_group_identity_and_ties(self):
        a=dict(task_id='59',policy_version=0,episode_id='a',complete=True,reward=1.)
        b=dict(a,episode_id='b',reward=0.)
        self.assertAlmostEqual(group_advantages([a,b])[0],1.)
        self.assertEqual(group_advantages([a,dict(a,episode_id='b')]),[0.,0.])
        for key,value in [('task_id','54'),('policy_version',1),('episode_id','a')]:
            with self.assertRaises(ValueError):group_advantages([a,dict(b,**{key:value})])

    def test_sampler_correction_preserves_unclipped_importance_ratio(self):
        import torch
        from tau2_grpo.rl_objective import clipped_token_loss
        new=torch.tensor([-2.1,-1.9],requires_grad=True)
        replay_old=torch.tensor([-2.,-2.])
        sampled=torch.tensor([-2.2,-1.8])
        loss=clipped_token_loss(new,replay_old,new.detach(),1.,0.2,0.,sampled)
        self.assertTrue(torch.allclose(loss,-torch.exp(new-sampled)))

    def test_citation_normalization_cannot_repair_invented_item_values(self):
        from tau2_grpo.rl_context import normalize_provenance
        from tau2_grpo.retail_memory import summary_errors
        raw=[{'role':'system','content':'policy'},{'role':'user','content':'help'},
             {'role':'assistant','content':'call','tool_calls':[{'function':{'name':'get_order_details','arguments':{'order_id':'#W1'}}}]},
             {'role':'tool','content':json.dumps({'name':'get_order_details','content':{'order_id':'#W1','status':'pending','items':[{'item_id':'i1','product_id':'p1','price':3.}]}})}]
        memory={'customer':{'user_id':None,'authenticated':True,'source_ids':[]},
                'active_order':{'order_id':'#W1','status':'pending','source_ids':[2]}}
        normalized,corrections=normalize_provenance(memory,raw)
        self.assertFalse(normalized['customer']['authenticated'])
        self.assertEqual(normalized['active_order']['source_ids'],[3])
        self.assertEqual(summary_errors(normalized,raw),[])
        normalized['items']=[{'item_id':'i1','product_id':'p1','price':999.,'source_ids':[2]}]
        normalized,_=normalize_provenance(normalized,raw)
        self.assertTrue(summary_errors(normalized,raw))

    def test_memory_unknown_values_and_fabricated_authentication(self):
        from tau2_grpo.retail_memory import summary_errors
        raw=[{'role':'system','content':'policy'},{'role':'user','content':'help'}]
        self.assertEqual(summary_errors({'customer':{'user_id':None,'authenticated':False,'source_ids':[]}},raw),[])
        self.assertTrue(summary_errors({'customer':{'user_id':'invented','authenticated':True,'source_ids':[1]}},raw))
        self.assertTrue(summary_errors({'customer':{'user_id':'future','source_ids':[2]}},raw))

    def test_real_lora_grpo_update_keeps_reference_frozen(self):
        import torch
        from peft import LoraConfig,get_peft_model
        from transformers import Cohere2Config,Cohere2ForCausalLM
        from tau2_grpo.rl_policy import LocalPolicy,disable_dropout
        torch.manual_seed(123)
        model=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            sliding_window=16,sliding_window_pattern=2,attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))
        lora=LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,task_type='CAUSAL_LM')
        model=get_peft_model(model,lora,adapter_name='policy');model.add_adapter('reference',lora)
        model.set_adapter('policy');disable_dropout(model)
        policy=LocalPolicy.__new__(LocalPolicy);policy.model=model;policy.has_reference=True;policy.version=0
        policy.cfg=replace(RetailRLConfig(),device='cpu',precision='fp32',replay_tolerance=1e-4)
        policy.tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0,bos_token_id=1,decode=lambda x,**kw:str(x.tolist()))
        policy.freeze_reference()
        before={n:p.detach().clone() for n,p in model.named_parameters() if '.policy.' in n or '.reference.' in n}
        episodes=[]
        for i,reward in enumerate([1.,0.]):
            prompt=[1,4+i,6]
            action=dict(prompt_ids=prompt,**policy.generate({'prompt_ids':prompt},4,123+i),policy_version=0)
            action['reference_logprobs']=policy.score(action,reference=True).detach().tolist()
            episodes.append(dict(task_id='59',policy_version=0,episode_id=str(i),complete=True,reward=reward,actions=[action]))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        result=update_group(policy,optimizer,episodes,policy.cfg)
        self.assertEqual(result['optimizer_steps'],1);self.assertEqual(policy.version,1)
        self.assertLess(result['max_policy_replay_error'],1e-4)
        changed=[]
        for n,p in model.named_parameters():
            if '.reference.' in n:
                self.assertFalse(p.requires_grad);self.assertTrue(torch.equal(before[n],p))
            elif '.policy.' in n:changed.append(not torch.equal(before[n],p))
        self.assertTrue(any(changed))

    def test_segmentation_does_not_change_episode_weight_or_gradients(self):
        import torch
        cfg=replace(RetailRLConfig(),kl_beta=0.,replay_tolerance=0.01)
        def run(split):
            parameter=torch.nn.Parameter(torch.tensor(0.))
            class Policy:
                version=0
                def score(self,a,grad):return parameter.expand(len(a['completion_ids']))*a['feature']
            def action(n,f):return dict(completion_ids=[1]*n,old_logprobs=[0.]*n,reference_logprobs=[0.]*n,feature=f,policy_version=0)
            episodes=[dict(task_id='59',episode_id='a',policy_version=0,complete=True,reward=1.,actions=[action(n,2.) for n in split]),
                      dict(task_id='59',episode_id='b',policy_version=0,complete=True,reward=0.,actions=[action(2,1.)])]
            opt=torch.optim.SGD([parameter],lr=0.1)
            metrics=update_group(Policy(),opt,episodes,cfg)
            self.assertEqual(metrics['max_policy_replay_error'],0.)
            return parameter.item()
        self.assertAlmostEqual(run([4]),run([1,3]),places=7)
        self.assertGreater(run([4]),0.)

    def test_tiny_model_sampled_vs_replay_logprobs_and_real_gradients(self):
        import torch
        from transformers import Cohere2Config,Cohere2ForCausalLM
        from tau2_grpo.rl_policy import LocalPolicy,action_logprobs,disable_dropout
        torch.manual_seed(0)
        model=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            sliding_window=16,sliding_window_pattern=2,attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))
        disable_dropout(model);model.eval()
        policy=LocalPolicy.__new__(LocalPolicy);policy.cfg=replace(RetailRLConfig(),device='cpu',precision='fp32',score_chunk_size=2)
        policy.model=model;policy.tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0,bos_token_id=1,decode=lambda x,**kw:str(x.tolist()))
        prompt=[1,4,5];generated=policy.generate({'prompt_ids':prompt},8,123)
        actual=action_logprobs(model,prompt,generated['completion_ids'],policy.cfg.temperature,2)
        expected=torch.tensor(generated['old_logprobs'])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5),(actual-expected).abs().max())
        before=model.lm_head.weight.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
        optimizer.zero_grad();(-actual.mean()).backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
        optimizer.step();self.assertFalse(torch.equal(before,model.lm_head.weight))
        # Nucleus sampling uses the requested support/normalization, not top_p=1.
        policy.cfg=replace(policy.cfg,temperature=0.7,top_p=0.95)
        nucleus=policy.generate({'prompt_ids':prompt},8,123)
        full=policy.score(dict(prompt_ids=prompt,**nucleus))
        gap=torch.tensor(nucleus['old_logprobs'])-full
        self.assertTrue(torch.isfinite(gap).all())
        self.assertGreater(float(gap.max()),0.)
        self.assertGreaterEqual(float(gap.min()),-1e-5)
        # Temperature zero means actual greedy inference, without dividing logits by zero.
        policy.cfg=replace(policy.cfg,temperature=0.)
        first=policy.generate({'prompt_ids':prompt},8,123)
        second=policy.generate({'prompt_ids':prompt},8,999)
        self.assertEqual(first['completion_ids'],second['completion_ids'])
        self.assertEqual(first['decoding_mode'],'greedy')
        self.assertEqual(first['old_logprobs'],[0.]*len(first['completion_ids']))
        scored=policy.score(dict(prompt_ids=prompt,**first))
        self.assertTrue(torch.isfinite(scored).all())
        self.assertTrue(torch.allclose(scored,torch.tensor(first['model_token_logprobs']),atol=1e-5))

    def test_real_tau2_reset_tool_bridge_and_terminal_reward_without_network(self):
        from loguru import logger
        logger.remove()
        from tau2_grpo.rl_tau2 import RetailEnvironment
        from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
        from tau2.registry import registry
        from tau2.gym.gym_agent import AgentGymEnv  # Registers Tau2's built-in Gym user.
        original=NLAssertionsEvaluator.__dict__['evaluate_nl_assertions'];user_count=len(registry._users)
        test_case=self
        class External:
            cfg=RetailRLConfig();audit=[]
            def request(self,messages,max_tokens,purpose,seed,json_mode=False):
                if purpose=='nl_judge':
                    payload=json.loads(messages[1]['content'])
                    assertions=payload['expectedOutcomes']
                    test_case.assertTrue(payload['conversation'])
                    test_case.assertTrue(all('raw_data' not in m and 'timestamp' not in m for m in payload['conversation']))
                    test_case.assertTrue(any(m['role']=='tool' and 'content' in m for m in payload['conversation']))
                    return json.dumps({'results':[dict(expectedOutcome=a,metExpectation=False,reasoning='Test stub') for a in assertions]})
                return 'Hello, I need help with my order.'
        env=RetailEnvironment('59',42,RetailRLConfig(),External())
        try:
            messages=env.reset();self.assertEqual(messages[0]['role'],'system')
            self.assertNotIn('evaluation_criteria',messages[0]['content'])
            result,done,call=env.step('<tool_call>{"name":"get_user_details","arguments":{"user_id":"invalid_user_for_test"}}</tool_call>')
            self.assertFalse(done);self.assertEqual(result[0]['role'],'tool')
            self.assertEqual(json.loads(result[0]['content'])['name'],'get_user_details')
            reward,simulation=env.finish();self.assertIn(reward,[0.,1.]);self.assertTrue(simulation['reward_info'])
            with self.assertRaises(RuntimeError):env.finish()
        finally:env.close()
        self.assertIs(NLAssertionsEvaluator.__dict__['evaluate_nl_assertions'],original)
        self.assertEqual(len(registry._users),user_count)


if __name__=='__main__':unittest.main()
