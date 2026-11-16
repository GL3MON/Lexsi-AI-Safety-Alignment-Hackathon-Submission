import importlib.util
import json
import unittest

from tau2_grpo.prepare_function_calling import (
    assign_splits, model_messages, normalize, parse_calls, prompt_group, rl_decisions,
)


class FunctionCallingDataTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema not installed")
    def test_embedded_catalog_and_argument_validation(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"}}, "required": ["number"]},
        }}]
        row = {"id": "source", "tools": json.dumps(tools), "conversations": [
            {"from": "system", "value": "Signatures are within <tools> </tools> tags.\n<tools>" + json.dumps(tools) + "</tools>"},
            {"from": "human", "value": "Look up number 1"},
            {"from": "gpt", "value": '<tool_call>{"name":"lookup","arguments":{"number":1}}</tool_call>'},
        ]}
        self.assertEqual(normalize(row, "test")["tool_calls"], 1)
        row["conversations"][-1]["value"] = '<tool_call>{"name":"lookup","arguments":{"number":"one"}}</tool_call>'
        with self.assertRaisesRegex(ValueError, "schema_invalid_arguments"):
            normalize(row, "test")

    def test_related_requests_share_group(self):
        first = [{"role": "user", "content": " Find   my order "}]
        extended = [{"role": "user", "content": "find my ORDER"},
                    {"role": "assistant", "content": "extra response"}]
        self.assertEqual(prompt_group(first), prompt_group(extended))

    def test_groups_stay_together(self):
        records = [
            {"group_id": str(i), "turn_type": turn_type}
            for i in range(20) for turn_type in ("single_turn", "multi_turn")
        ]
        assignment = assign_splits(records)
        self.assertEqual(assignment, assign_splits(list(reversed(records))))
        self.assertEqual(list(assignment.values()).count("rl"), 4)

    def test_multiple_calls_and_invalid_targets(self):
        text = '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        self.assertEqual(len(parse_calls(text + text)), 2)
        for bad in [
            '<tool_call>{"name":"a","arguments":"{}"}</tool_call>',
            '<tool_call>{"name":"a","arguments_json":{}}</tool_call>',
            '<tool_call>{"name":"a","arguments":{}}',
            "<tool_call>{'name':'a','arguments':{}}</tool_call>",
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_calls(bad)

    def test_rl_reference_is_not_in_prompt(self):
        record = {
            "id": "c1", "group_id": "g1", "turn_type": "multi_turn", "tools": [],
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'},
                {"role": "tool", "content": "<tool_response>result</tool_response>"},
                {"role": "assistant", "content": "final answer"},
            ],
        }
        decisions = list(rl_decisions(record))
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["prompt"], record["messages"][:2])
        self.assertEqual(decisions[1]["prompt"][-1]["role"], "user")
        self.assertNotIn("final answer", str(decisions[1]["prompt"]))
        self.assertEqual(decisions[1]["reward_reference"]["assistant"], "final answer")
        self.assertEqual(record["messages"][3]["role"], "tool")
        self.assertEqual(model_messages(record["messages"])[3]["role"], "user")
