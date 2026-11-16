import importlib.util
import unittest
from types import SimpleNamespace
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "checkpoint_evaluation", Path(__file__).resolve().parents[1] / "scripts/eval_sft_checkpoint.py"
)
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema not installed")
class ToolEvaluationTests(unittest.TestCase):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"],
    }}}]
    expected = [{"name": "lookup", "arguments": {"id": "a"}}]

    def test_native_and_training_stop_tokens_are_both_respected(self):
        tokenizer = SimpleNamespace(eos_token_id=6, unk_token_id=0,
                                    convert_tokens_to_ids=lambda token: 255001)
        self.assertEqual(EVALUATION.generation_stop_ids(255001, tokenizer), [6, 255001])
        self.assertEqual(EVALUATION.generation_stop_ids([255001, 3], tokenizer), [3, 6, 255001])

    def test_correct_call(self):
        result = EVALUATION.score('<tool_call>{"name":"lookup","arguments":{"id":"a"}}</tool_call>', self.expected, self.tools)
        self.assertTrue(result["exact_calls"])
        self.assertTrue(result["schema_valid"])

    def test_well_formed_wrong_argument_is_not_correct(self):
        result = EVALUATION.score('<tool_call>{"name":"lookup","arguments":{"id":"wrong"}}</tool_call>', self.expected, self.tools)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["exact_calls"])

    def test_schema_and_tool_name_errors(self):
        for text in [
            '<tool_call>{"name":"lookup","arguments":{"id":123}}</tool_call>',
            '<tool_call>{"name":"unknown","arguments":{"id":"a"}}</tool_call>',
        ]:
            result = EVALUATION.score(text, self.expected, self.tools)
            self.assertTrue(result["syntax_valid"])
            self.assertFalse(result["schema_valid"])
            self.assertFalse(result["exact_calls"])

    def test_missing_or_malformed_call(self):
        for text in ["I'll look that up", '<tool_call>{"name":"lookup","arguments_json":"{}"}</tool_call>']:
            result = EVALUATION.score(text, self.expected, self.tools)
            self.assertFalse(result["syntax_valid"])
            self.assertFalse(result["exact_calls"])

    def test_followup_no_call_is_separate_from_text_correctness(self):
        result = EVALUATION.score("Here is the result", [], self.tools)
        self.assertTrue(result["exact_calls"])
        result = EVALUATION.score('<tool_call>{"name":"lookup","arguments":{"id":"a"}}</tool_call>', [], self.tools)
        self.assertFalse(result["exact_calls"])
