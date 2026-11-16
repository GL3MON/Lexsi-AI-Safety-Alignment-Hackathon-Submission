import json
import unittest
from tau2_grpo.glaive_prompts import check_system,system_prompt
from tau2_grpo.prepare_function_calling import parse_calls
from scripts.build_clean_tool_cases import build,tools

class CleanPromptTests(unittest.TestCase):
    def test_single_catalog_with_exact_schemas(self):
        catalog=tools();text=system_prompt(catalog);check_system(text,catalog)
        self.assertEqual(text.count('<tools>'),1)
        with self.assertRaises(ValueError):check_system(text+text,catalog)
    def test_twenty_fixed_ascii_argument_cases(self):
        cases=build();self.assertEqual(len(cases),20)
        self.assertEqual(len({c['id'] for c in cases}),20)
        for c in cases:
            self.assertNotIn('translate_text',json.dumps(c))
            self.assertTrue(c['manual_hint'])
            for i,m in enumerate(c['prompt'][1:]):self.assertEqual(m['role'],'user' if i%2==0 else 'assistant')
    def test_escape_representation_does_not_change_value(self):
        c={'name':'x','arguments':{'text':'très'}}
        a=parse_calls('<tool_call>'+json.dumps(c,ensure_ascii=True)+'</tool_call>')
        b=parse_calls('<tool_call>'+json.dumps(c,ensure_ascii=False)+'</tool_call>')
        self.assertEqual(a,b)
        d={'name':'x','arguments':{'text':r'tr\u00e8s'}}
        self.assertNotEqual(a,parse_calls('<tool_call>'+json.dumps(d)+'</tool_call>'))
