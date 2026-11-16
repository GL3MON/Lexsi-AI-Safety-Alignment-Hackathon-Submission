import unittest
from tau2_grpo.glaive_rewards import score_calls


def call(name='lookup', **args):
    return {'name': name, 'arguments': args}


def output(*calls):
    import json
    return ''.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in calls)


class GlaiveRewardTests(unittest.TestCase):
    def test_correct_and_extra(self):
        gold=call(email='A.b+1@example.org')
        self.assertEqual(score_calls(output(gold),[gold])['reward'],1)
        self.assertEqual(score_calls(output(gold,call(email='wrong')),[gold])['reward'],.5)

    def test_duplicate_cannot_farm_reward(self):
        gold=call(zip='02139')
        self.assertEqual(score_calls(output(gold,gold),[gold])['reward'],.5)

    def test_exact_strings_and_types(self):
        for wrong in [call(zip='2139'),call(zip=2139),call(zip='02139.001')]:
            self.assertEqual(score_calls(output(wrong),[call(zip='02139')])['reward'],0)
        self.assertEqual(score_calls(output(call(x=True)),[call(x=1)])['reward'],0)

    def test_no_call_and_malformed(self):
        gold=call(x='a')
        for text in ['Could you clarify?',output(gold)+'<tool_call>{oops}</tool_call>',output(gold)+'<tool_call>',output(gold)+'<tool_response>fake</tool_response>']:
            self.assertEqual(score_calls(text,[gold])['reward'],0)

    def test_precision_alone_does_not_measure_coverage(self):
        a,b=call(x='a'),call(x='b')
        result=score_calls(output(a),[a,b])
        self.assertEqual(result['reward'],1)
        self.assertEqual(result['call_recall'],.5)
        self.assertFalse(result['exact_call_set'])

    def test_order_of_calls_and_object_keys_does_not_matter(self):
        a,b=call(x='a'),call(x='b')
        self.assertEqual(score_calls(output(b,a),[a,b])['reward'],1)
