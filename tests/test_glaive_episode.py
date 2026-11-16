import json
import unittest
from types import SimpleNamespace
from tau2_grpo.glaive_episode import rollout,episode_rows,tool_feedback


class Tokenizer:
    def __init__(self):self.contexts=[]
    def apply_chat_template(self,messages,**kwargs):
        self.contexts.append(json.loads(json.dumps(messages)));return [1,2]


class Policy:
    version=0
    def __init__(self,outputs):self.outputs=iter(outputs);self.tokenizer=Tokenizer()
    def generate(self,*args):return {'text':next(self.outputs),'ended_with_eos':True,'completion_ids':[3],'old_logprobs':[-1.]}


def fixture():
    c={'name':'lookup','arguments':{'email':'N_Varela+orders@post.example.org'}}
    return {'id':'one','source_id':'src','group_id':'g','scenario':'missing_required','prompt':[{'role':'system','content':'rules'},{'role':'user','content':'I need help but have not provided email.'}], 'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'email':{'type':'string'}},'required':['email']}}}], 'reward_reference':{'decision':'clarify','missing_field_aliases':['email']},'customer_reply':{'role':'user','content':'My email is N_Varela+orders@post.example.org.'},'final_reference':{'tool_calls':[c],'literal_arguments':{'lookup':c['arguments']}}}


class EpisodeTests(unittest.TestCase):
    cfg=SimpleNamespace(max_actions=10,response_tokens=32,context_limit=128)
    def test_customer_replies_and_actual_history_is_replayed(self):
        row=fixture();call='<tool_call>'+json.dumps(row['final_reference']['tool_calls'][0])+'</tool_call>'
        policy=Policy(['Please provide your email.',call]);e=rollout(policy,self.cfg,row,42,'e')
        self.assertEqual(e['reward'],1)
        self.assertEqual(e['agent_turns'],2)
        self.assertTrue(e['customer_replied'])
        self.assertEqual(policy.tokenizer.contexts[1][-2]['content'],'Please provide your email.')
        self.assertEqual(policy.tokenizer.contexts[1][-1],row['customer_reply'])
        self.assertNotIn('N_Varela',str(policy.tokenizer.contexts[0]))

    def test_turn_hints_are_current_only_and_never_modify_runtime_history(self):
        row=fixture();call='<tool_call>'+json.dumps(row['final_reference']['tool_calls'][0])+'</tool_call>'
        p=Policy(['Please provide your email.',call])
        e=rollout(p,self.cfg,row,42,'e',turn_hints={1:'FIRST_PRIVATE_HINT',2:'SECOND_PRIVATE_HINT'})
        self.assertIn('FIRST_PRIVATE_HINT',p.tokenizer.contexts[0][0]['content'])
        self.assertNotIn('SECOND_PRIVATE_HINT',p.tokenizer.contexts[0][0]['content'])
        self.assertIn('SECOND_PRIVATE_HINT',p.tokenizer.contexts[1][0]['content'])
        self.assertNotIn('FIRST_PRIVATE_HINT',p.tokenizer.contexts[1][0]['content'])
        self.assertNotIn('PRIVATE_HINT',str(e['messages']))
        self.assertEqual(e['reward'],1.)

    def test_premature_call_does_not_unlock_customer_value(self):
        e=rollout(Policy(['<tool_call>{"name":"lookup","arguments":{"email":""}}</tool_call>']),self.cfg,fixture(),42,'e')
        self.assertEqual(e['reward'],0)
        self.assertFalse(e['customer_replied'])
        self.assertEqual(len(e['actions']),1)

    def test_repeated_question_after_customer_reply_gets_no_call_credit(self):
        e=rollout(Policy(['What is your email?','What is your email?']),self.cfg,fixture(),42,'e')
        self.assertEqual(e['reward'],.25)
        self.assertEqual(e['termination_reason'],'incorrect_final_decision')

    def test_context_failure_is_not_negative_training_example(self):
        cfg=SimpleNamespace(max_actions=10,response_tokens=32,context_limit=1)
        e=rollout(Policy([]),cfg,fixture(),42,'e')
        self.assertFalse(e['complete']);self.assertIsNone(e['reward'])

    def test_followup_not_trained_twice(self):
        row=fixture();follow={**row,'id':'follow','scenario':'clarification_followup','prompt':row['prompt']+[{'role':'assistant','content':'What is your email?'},row['customer_reply']],'reward_reference':row['final_reference']}
        result=episode_rows([row,follow]);self.assertEqual(len(result),1)
        self.assertEqual(result[0]['customer_reply'],row['customer_reply'])

    def test_tool_success_only_after_exact_valid_call(self):
        from tau2_grpo.glaive_signals import score_decision
        row=fixture();ref={'tools':row['tools'],'reward_reference':row['final_reference']}
        text='<tool_call>'+json.dumps(row['final_reference']['tool_calls'][0])+'</tool_call>'
        details=score_decision(text,ref)
        response=tool_feedback(text,row,details)
        self.assertIn('"call_verified": true',response['content'])
        self.assertIn('"api_executed": false',response['content'])
        wrong='<tool_call>{"name":"lookup","arguments":{"email":"wrong"}}</tool_call>'
        response=tool_feedback(wrong,row,score_decision(wrong,ref))
        self.assertIn('call_failed_verification',response['content'])
        self.assertNotIn('reference_call_matches',response['content'])
        self.assertIsNone(tool_feedback('Please provide your email.',row,{}))

    def test_turn_cap_prevents_second_generation(self):
        cfg=SimpleNamespace(max_actions=1,response_tokens=32,context_limit=128)
        e=rollout(Policy(['What is your email?']),cfg,fixture(),42,'e')
        self.assertEqual(e['agent_turns'],1)
        self.assertFalse(e['customer_replied'])
