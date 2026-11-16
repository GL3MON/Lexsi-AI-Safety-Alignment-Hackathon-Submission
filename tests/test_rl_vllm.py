import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tau2_grpo.rl_config import RetailRLConfig, load_settings
from tau2_grpo.rl_vllm import VLLMPolicy, sync_adapter


class VLLMTests(unittest.TestCase):
    def sampler(self, tokens, logs):
        self.request = None
        def create(**kwargs):
            self.request = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(
                tokens=tokens, token_logprobs=logs))])
        tokenizer = SimpleNamespace(eos_token_id=6, decode=lambda ids, **kw: str(ids))
        return VLLMPolicy(RetailRLConfig(), 3, tokenizer,
                          SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_exact_token_ids_probabilities_and_trained_eos(self):
        result = self.sampler(['token_id:44', 'token_id:6'], [-1.2, -.2]).generate(
            {'prompt_ids': [1, 99]}, 50, 17)
        self.assertEqual(result['completion_ids'], [44, 6])
        self.assertEqual(result['old_logprobs'], [-1.2, -.2])
        self.assertTrue(result['ended_with_eos'])
        self.assertEqual(self.request['prompt'], [1, 99])
        self.assertEqual(self.request['model'], 'policy_v3')
        self.assertEqual(self.request['extra_body']['stop_token_ids'], [6])
        self.assertTrue(self.request['extra_body']['ignore_eos'])

    def test_rejects_missing_or_decoded_probabilities(self):
        for tokens, logs in [(['hello'], [-1.]), (['token_id:2'], [None]),
                             (['token_id:2'], [float('nan')]), ([], [])]:
            with self.subTest(tokens=tokens, logs=logs), self.assertRaises(RuntimeError):
                self.sampler(tokens, logs).generate({'prompt_ids': [1]}, 10, 1)
        result = self.sampler(['token_id:44'], [-1.]).generate({'prompt_ids': [1]}, 1, 1)
        self.assertFalse(result['ended_with_eos'])

    def test_adapter_replace_order_and_unique_versions(self):
        def save(destination, **kwargs):
            folder = destination / 'policy';folder.mkdir(parents=True)
            (folder/'adapter_config.json').write_text('{}')
        policy = SimpleNamespace(version=1, model=SimpleNamespace(save_pretrained=save))
        with tempfile.TemporaryDirectory() as temp, patch('tau2_grpo.rl_vllm.adapter_request') as send:
            sync_adapter(policy, RetailRLConfig(), temp)
            self.assertEqual([c.args[1] for c in send.call_args_list],
                             ['unload_lora_adapter', 'load_lora_adapter'])
            self.assertEqual(send.call_args_list[0].args[2]['lora_name'], 'policy_v0')
            self.assertEqual(send.call_args_list[1].args[2]['lora_name'], 'policy_v1')

    def test_hf_uses_eight_rollouts_and_checkpoint_100(self):
        cfg = load_settings('hf_jobs/retail_rl/smoke-config.json')
        self.assertEqual((cfg.group_size, cfg.rollout_workers), (8, 8))
        self.assertEqual(cfg.rollout_engine, 'transformers')
        self.assertTrue(cfg.initial_adapter.endswith('checkpoint-100'))


if __name__ == '__main__':
    unittest.main()
