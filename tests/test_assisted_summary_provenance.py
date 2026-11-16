import unittest,json
from pathlib import Path
from tau2_grpo.rl_context import normalize_provenance
from tau2_grpo.retail_memory import summary_errors,controller_history
class ProvenanceTests(unittest.TestCase):
 def test_recorded_failed_summaries_revalidate_without_changing_facts(self):
  root=Path('outputs/hf-jobs/6afa120923985fb9cd83af27/bucket/tau2-assisted-rl-20261115T201153/smoke/group-0000');checked=0
  for file in root.glob('episode-*.json'):
   episode=json.loads(file.read_text())
   if not episode.get('error','').startswith('ContextError:'):continue
   raw=episode['messages']
   for audit in episode['summary_attempts']:
    if not any('payment_method_id' in e for e in audit.get('validation_errors',[])):continue
    memory=json.loads(audit['response']['choices'][0]['message']['content']);before=memory.get('payment',{}).get('payment_method_id');fixed,changes=normalize_provenance(memory,raw);self.assertEqual(fixed.get('payment',{}).get('payment_method_id'),before)
    payload=json.loads(audit['request'][1]['content']);tail=payload['recent_tail_message_ids'];fixed.pop('confirmation',None);fixed['observed_history']=controller_history(raw);fixed['recent_exchange']=dict(source_ids=tail,owner='controller',instruction='Read the original turns')
    self.assertEqual(summary_errors(fixed,raw),[],(file.name,changes));checked+=1
  self.assertGreaterEqual(checked,7)
 def test_wrong_tool_pair_is_not_remapped(self):
  raw=[dict(role='system',content='s'),dict(role='assistant',content='<tool_call>{"name":"get_user_details","arguments":{"user_id":"u"}}</tool_call>'),dict(role='tool',content='{"name":"get_order_details","content":{}}')]
  memory={'payment':{'payment_method_id':'invented','source_ids':[1]}};fixed,changes=normalize_provenance(memory,raw);self.assertEqual(fixed,memory);self.assertEqual(changes,[])
 def test_malformed_call_is_not_provenance(self):
  raw=[dict(role='system',content='s'),dict(role='assistant',content='<tool_call>{broken}</tool_call>'),dict(role='tool',content='{"name":"get_user_details","content":{}}')]
  memory={'payment':{'payment_method_id':'invented','source_ids':[1]}};fixed,_=normalize_provenance(memory,raw);self.assertEqual(fixed,memory)
