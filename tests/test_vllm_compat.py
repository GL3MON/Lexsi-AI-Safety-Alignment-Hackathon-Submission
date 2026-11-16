import ast
import unittest
from types import SimpleNamespace

from tau2_grpo.vllm_compat import patch_cohere_lora


SOURCE = '''
def compute_logits(self, hidden_states):
    is_not_lora = hasattr(self.model.embed_tokens, "weight")
    if is_not_lora:
        return self.logits_processor(self.model.embed_tokens, hidden_states)
    return self.logits_processor(self.model.embed_tokens.base_layer, hidden_states)
'''


class CompatibilityTests(unittest.TestCase):
    def test_wrapped_embedding_forwarding_weight_uses_base_for_logits(self):
        scope = {}
        exec(patch_cohere_lora(SOURCE), scope)
        bare = SimpleNamespace(weight=object(), quant_method=object())
        wrapped = SimpleNamespace(weight=bare.weight, base_layer=bare)
        for embedding in [bare, wrapped]:
            actor = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding),
                logits_processor=lambda head, hidden: (head.quant_method, hidden))
            result = scope['compute_logits'](actor, 'hidden')
            self.assertEqual(result, (bare.quant_method, 'hidden'))

    def test_idempotent_and_rejects_unexpected_source(self):
        updated = patch_cohere_lora(SOURCE)
        self.assertEqual(patch_cohere_lora(updated), updated)
        with self.assertRaises(RuntimeError): patch_cohere_lora('different source')


if __name__ == '__main__':
    unittest.main()
