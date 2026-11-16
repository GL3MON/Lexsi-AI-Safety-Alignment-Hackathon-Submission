from dataclasses import replace
from types import SimpleNamespace
import unittest

from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_transformers_workers import serve_requests


class TransformersWorkerTests(unittest.TestCase):
    def test_broker_preserves_each_output_budget_and_result_order(self):
        calls = []
        def generate(prepared, max_tokens, seed):
            calls.append((len(prepared), max_tokens, seed))
            return [{'value': p['prompt_ids']} for p in prepared]
        outputs = [[], [], []]
        requests = [(SimpleNamespace(send=out.append), {'prompt_ids':[i]}, limit, 42+i)
                    for i,(out,limit) in enumerate(zip(outputs,[20,10,20]))]
        serve_requests(SimpleNamespace(generate_batch=generate), requests)
        self.assertEqual(calls,[(2,20,42),(1,10,43)])
        self.assertEqual([out[0][1]['value'] for out in outputs],[[0],[1],[2]])

    def test_real_cohere_greedy_batch_matches_individual_with_left_padding(self):
        import torch
        from transformers import Cohere2Config, Cohere2ForCausalLM
        from tau2_grpo.rl_policy import LocalPolicy, disable_dropout
        torch.set_num_threads(1);torch.manual_seed(0)
        model=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            sliding_window=16,sliding_window_pattern=2,attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))
        disable_dropout(model)
        policy=LocalPolicy.__new__(LocalPolicy)
        policy.cfg=replace(RetailRLConfig(),device='cpu',precision='fp32',temperature=0.)
        policy.model=model
        policy.tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0,bos_token_id=1,
            decode=lambda ids,**kw:str(list(ids)))
        prompts=[{'prompt_ids':[1,4,5]},{'prompt_ids':[1,3,4,5]*6}]
        expected=[policy.generate(p,6,42)['completion_ids'] for p in prompts]
        actual=policy.generate_batch(prompts,6,42)
        self.assertEqual([r['completion_ids'] for r in actual],expected)
        self.assertTrue(all(r['decoding_mode']=='greedy' for r in actual))
        self.assertTrue(all(r['generation_batch_size']==2 for r in actual))


if __name__=='__main__':unittest.main()
