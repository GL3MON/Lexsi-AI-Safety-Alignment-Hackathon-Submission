"""Replay and independently re-encode the complete exported next-action dataset."""
import json,hashlib
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from tau2.domains.retail.data_model import get_db
from tau2_grpo.sft_data import read_records,encode_records
from scripts.generate_retail_next_action_sft import assemble,check_text,ROOT


def main():
 out=ROOT/'outputs/retail-data/retail-next-action-480';rows=read_records(out/'all.jsonl');train=read_records(out/'sft.jsonl');val=read_records(out/'validation.jsonl')
 assert len(rows)==480 and len(train)==432 and len(val)==48
 assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
 assert all(n==2 for n in Counter(r['group_id'] for r in rows).values())
 base=get_db();system=rows[0]['messages'][0]['content'];calls=0
 for r in rows:
  n=int(r['id'].rsplit('-',1)[-1]);_,prefix,audit,expected,instruction,profile,known,split,family=assemble(base,n,system)
  assert r['messages'][:-1]==prefix,(r['id'],'prefix mismatch')
  assert r['tool_execution_audit']==audit,(r['id'],'tool replay mismatch')
  assert (r['profile'],r['order_id_known'],r['split'])==(profile,known,split)
  assert r['loss_message_indices']==[len(r['messages'])-1]
  assert r['expected_call']==expected
  target=r['messages'][-1]['content'];check_text(target.split('<tool_call>',1)[0].strip(),instruction,prefix)
  if expected:
   actual=json.loads(target.split('<tool_call>',1)[1].split('</tool_call>',1)[0]);assert actual==expected
  else:assert '<tool_call>' not in target
  calls+=len(audit)
 tok=AutoTokenizer.from_pretrained(ROOT/'outputs/hf-jobs/6af8591d23985fb9cd820ff6/eval-snapshot/training/run-2a32064432/checkpoint-50');tok.chat_template=(ROOT/'templates/retail_sdg_v2.jinja').read_text()
 count=0;maximum=0;supervised=0
 for row,enc,spans,error in encode_records(tok,rows,8192,1):
  assert not error,(row['id'],error)
  selected=set();indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
  for i,(start,end) in zip(indices,spans,strict=True):
   if i==len(row['messages'])-1:
    assert enc['labels'][start:end]==enc['input_ids'][start:end];selected.update(range(start,end))
   else:assert all(v==-100 for v in enc['labels'][start:end])
  assert all(v==-100 for i,v in enumerate(enc['labels']) if i not in selected)
  assert len(enc['input_ids'])==row['context_tokens']
  count+=1;maximum=max(maximum,len(enc['input_ids']));supervised+=len(selected)
  if count%100==0:print('AUDITED',count,flush=True)
 report=dict(records=count,train=len(train),validation=len(val),all_passed=True,duplicates=0,train_validation_group_overlap=0,paired_families=len(Counter(r['group_id'] for r in rows)),real_tool_calls_replayed=calls,max_context_tokens=maximum,supervised_tokens=supervised,one_next_response_supervised=True,other_assistant_and_user_tool_messages_masked=True,files_sha256={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ['sft.jsonl','validation.jsonl','all.jsonl']})
 (out/'final-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
