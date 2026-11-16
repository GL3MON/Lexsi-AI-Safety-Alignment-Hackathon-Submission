import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tau2_grpo.sft_config import load_settings
from tau2_grpo.sft_data import split_validation, tokenize_assistant_only


class CharacterTemplate:
    """Prefix-stable chat fixture; tests labels independently of tokenizer internals."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        text = "".join(f"<{m['role']}>" + m["content"] + "<end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


class SFTTests(unittest.TestCase):
    def test_trl_collator_preserves_explicit_labels_and_masks_padding(self):
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        batch = collator([
            {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
            {"input_ids": [1, 8, 9], "labels": [-100, 8, 9]},
        ])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 3, 4], [-100, 8, 9, -100]])

    def test_chunked_forward_keeps_generation_capabilities(self):
        import inspect
        from types import MethodType, SimpleNamespace
        from tau2_grpo.sft_loss import preserve_generation_signature

        class Base:
            def forward(self, input_ids=None, logits_to_keep=0, **kwargs):
                return logits_to_keep

        base = Base()
        original = base.forward
        def _chunked_ce_forward(self, input_ids=None, labels=None, **kwargs):
            return original(input_ids=input_ids, **kwargs)
        base.forward = MethodType(_chunked_ce_forward, base)
        self.assertNotIn("logits_to_keep", inspect.signature(base.forward).parameters)
        preserve_generation_signature(SimpleNamespace(get_base_model=lambda: base))
        self.assertIn("logits_to_keep", inspect.signature(base.forward).parameters)
        self.assertEqual(base.forward(logits_to_keep=1), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_native_projection_loss_and_gradients(self):
        import torch
        from tau2_grpo.sft_loss import projection_chunk

        torch.manual_seed(1)
        h = torch.randn(4, 3, requires_grad=True)
        w = torch.randn(7, 3, requires_grad=True)
        b = torch.randn(7, requires_grad=True)
        labels = torch.tensor([0, 2, 3, 6])
        loss, correct, entropy = projection_chunk(h, w, b, labels, 0.5, 2.0)
        logits = 2 * torch.tanh((torch.nn.functional.linear(h, w, b) * 0.5) / 2)
        expected = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        self.assertTrue(torch.allclose(loss, expected))
        actual_grads = torch.autograd.grad(loss, (h, w, b), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (h, w, b))
        for actual, reference in zip(actual_grads, expected_grads):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
        self.assertEqual(correct.item(), (logits.argmax(-1) == labels).sum().item())
        probs = torch.softmax(logits, -1)
        self.assertTrue(torch.allclose(entropy, -(probs * probs.log()).sum()))

    def config(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            return load_settings(path)

    def test_wandb_required_and_types_checked(self):
        self.assertEqual(self.config({}).backend, "aligntune")
        self.assertEqual(self.config({"backend": "trl"}).backend, "trl")
        self.assertEqual(self.config({}).wandb_mode, "online")
        self.assertEqual(self.config({"wandb_mode": "offline"}).wandb_mode, "offline")
        for raw in [
            {"wandb_mode": "disabled"}, {"wandb_project": ""},
            {"batch_size": True}, {"learning_rate": 0}, {"typo": 1},
            {"precision": "unknown"}, {"use_cpu": True},
            {"max_train_samples": 0}, {"save_steps": 7, "eval_steps": 5},
            {"lora_target_modules": [1]},
            {"backend": "unknown"},
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.config(raw)

    def test_all_and_only_assistant_tokens_are_targets(self):
        # Repeated content tests that masking doesn't rely on searching for text.
        messages = [
            {"role": "system", "content": "repeat"},
            {"role": "user", "content": "repeat"},
            {"role": "assistant", "content": "repeat"},
            {"role": "tool", "content": "<tool_response>repeat</tool_response>"},
            {"role": "assistant", "content": "final"},
        ]
        encoded, spans = tokenize_assistant_only(CharacterTemplate(), messages, 1000)
        targets = [bytes(encoded["input_ids"][a:b]).decode() for a, b in spans]
        self.assertEqual(targets, ["repeat<end>", "final<end>"])
        for index, label in enumerate(encoded["labels"]):
            supervised = any(a <= index < b for a, b in spans)
            self.assertEqual(label, encoded["input_ids"][index] if supervised else -100)
        with self.assertRaisesRegex(ValueError, "overlength"):
            tokenize_assistant_only(CharacterTemplate(), messages, 5)

    def test_fail_on_template_that_rewrites_prefixes(self):
        class BrokenTemplate(CharacterTemplate):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                return [len(messages)] + ids

        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with self.assertRaisesRegex(ValueError, "prefix-preserving"):
            tokenize_assistant_only(BrokenTemplate(), messages, 1000)

    def test_validation_keeps_related_conversations_together(self):
        records = [
            {"group_id": str(group), "messages": [
                {"role": "user", "content": str(group)},
                {"role": "assistant", "content": str(variant)},
            ]}
            for group in range(20) for variant in range(2)
        ]
        train, evaluation = split_validation(records, 0.1, 42)
        self.assertEqual((train, evaluation), split_validation(records, 0.1, 42))
        self.assertFalse({r['group_id'] for r in train} & {r['group_id'] for r in evaluation})
        self.assertEqual(len(evaluation), 4)


if __name__ == "__main__":
    unittest.main()
