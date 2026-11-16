import unittest

from tau2_grpo.retail_preview import parse_call


class ToolFormatTests(unittest.TestCase):
    def test_direct_call(self):
        call = parse_call('<tool_call>{"name":"get_user_details","arguments":{"user_id":"u1"}}</tool_call>')
        self.assertEqual(call["arguments"], {"user_id": "u1"})

    def test_customer_message(self):
        self.assertIsNone(parse_call("What is your email address?"))

    def test_malformed_calls(self):
        for text in [
            '<tool_call>{"name":"done","arguments":"{}"}</tool_call>',
            '<tool_call>{"name":"done","arguments_json":"{}"}</tool_call>',
            '<tool_call>{"name":"done","arguments":{}}',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_call(text)
