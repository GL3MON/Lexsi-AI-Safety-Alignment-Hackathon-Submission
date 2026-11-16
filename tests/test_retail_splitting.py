import importlib.util
import unittest
from collections import Counter
from pathlib import Path

from test_sft_backends import ByteTokenizer

from tau2_grpo.sft_data import tokenize_assistant_only

spec = importlib.util.spec_from_file_location("retail_splitter", Path(__file__).resolve().parents[1] / "scripts/split_retail_sft_traces.py")
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


class RetailSplittingTests(unittest.TestCase):
    def test_overlap_does_not_duplicate_targets_or_split_tool_exchange(self):
        messages = [{"role": "system", "content": "policy"}]
        for i in range(3):
            messages.extend([{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "answer " * 5}])
        call_index = len(messages)
        messages += [
            {"role": "assistant", "content": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'},
            {"role": "user", "content": '<tool_response>{"name":"lookup","content":"found"}</tool_response>'},
            {"role": "assistant", "content": "finished"},
        ]
        count = lambda ms: sum(len(m["content"]) + 10 for m in ms)
        chunks = splitter.split_messages(messages, count, 180, 70)
        self.assertGreater(len(chunks), 1)
        trained = Counter()
        for chunk in chunks:
            self.assertLessEqual(count(chunk["messages"]), 180)
            self.assertEqual(chunk["messages"][0], messages[0])
            indices = chunk["source_message_indices"]
            self.assertEqual(call_index in indices, call_index + 1 in indices)
            trained.update(indices[i] for i in chunk["loss_message_indices"])
        self.assertEqual(trained, Counter(i for i, m in enumerate(messages) if m["role"] == "assistant"))

    def test_overlap_assistant_tokens_are_masked(self):
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "new response"},
        ]
        encoded, spans = tokenize_assistant_only(ByteTokenizer(), messages, 1000, [4])
        start, end = spans[-1]
        for i, label in enumerate(encoded["labels"]):
            self.assertEqual(label, encoded["input_ids"][i] if start <= i < end else -100)

    def test_exchange_too_large_fails_instead_of_cutting_json(self):
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "look up my order"},
            {"role": "assistant", "content": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'},
            {"role": "user", "content": '<tool_response>' + 'x' * 300 + '</tool_response>'},
        ]
        with self.assertRaisesRegex(ValueError, "single exchange"):
            splitter.split_messages(messages, lambda ms: sum(len(m["content"]) for m in ms), 100)
