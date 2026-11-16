import unittest
import json
from tau2_grpo.glaive_signals import score_decision


def row():
    call={'name':'lookup','arguments':{'email':'a.b+2@example.org','zip':'02139'}}
    return {'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'email':{'type':'string'},'zip':{'type':'string'}},'required':['email','zip']}}}], 'reward_reference':{'tool_calls':[call],'literal_arguments':{'lookup':call['arguments']}}}


def output(call):return '<tool_call>'+json.dumps(call)+'</tool_call>'


class SignalTests(unittest.TestCase):
    def test_exact_beats_partial_and_irrelevant(self):
        r=row();gold=r['reward_reference']['tool_calls'][0]
        good=score_decision(output(gold),r)
        bad=score_decision(output({'name':'lookup','arguments':{'email':'wrong','zip':'02139'}}),r)
        wrong=score_decision(output({'name':'other','arguments':{}}),r)
        self.assertEqual(good['reward'],1)
        self.assertGreater(bad['reward'],wrong['reward'])
        self.assertEqual(bad['copy_accuracy'],.5)

    def test_duplicate_reduces_reward(self):
        r=row();text=output(r['reward_reference']['tool_calls'][0])
        self.assertLess(score_decision(text+text,r)['reward'],score_decision(text,r)['reward'])

    def test_malformed_and_no_call_never_earn_dense_bonus(self):
        for text in ['Hello','<tool_call>{oops}</tool_call>', '<tool_call>{"name":"lookup","arguments":{"zip":"02139","zip":"02139"}}</tool_call>']:
            self.assertEqual(score_decision(text,row())['reward'],0)

    def test_unknown_argument_schema_invalid(self):
        r=row();c=r['reward_reference']['tool_calls'][0];c={**c,'arguments':{**c['arguments'],'bogus':'x'}}
        result=score_decision(output(c),r)
        self.assertEqual(result['schema_validity'],0)
        self.assertLess(result['reward'],1)

    def test_clarification_must_target_missing_field_without_call(self):
        r={'reward_reference':{'decision':'clarify','missing_field_aliases':['zip','postal code']}}
        self.assertEqual(score_decision('What is your ZIP?',r)['reward'],1)
        self.assertEqual(score_decision('Please provide your postal code.',r)['reward'],1)
        for text in ['What is your email?','ZIP code is 12345.', 'What is your ZIP?'+output({'name':'lookup','arguments':{}})]:
            self.assertEqual(score_decision(text,r)['reward'],0)
