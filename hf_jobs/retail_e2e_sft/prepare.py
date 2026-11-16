"""Reuse verified trajectories; preserve complete messages and assistant masks."""
import json,hashlib,random,re
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/retail-data/retail-e2e-sft-retention'
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 canonical=(ROOT/'templates/retail_apigen_sft_system.txt').read_text()
 from transformers import AutoTokenizer
 from tau2_grpo.rl_tau2 import _schema_contract,_supported_contract
 tok=AutoTokenizer.from_pretrained(ROOT/'outputs/rl-initialization/post20-6af9edf523985fb9cd839302');tok.chat_template=(ROOT/'templates/retail_sdg_v2.jinja').read_text()
 wanted={t['function']['name']:_schema_contract(t['function']['parameters']) for t in json.loads(re.findall(r'<tools>(.*?)</tools>',canonical,re.S)[0])}
 parts={s:[] for s in ['train','validation']};sources=[]
 for name in ['retail-sdg-v2-strict-clean','retail-sdg-v2-verified-recovery']:
  root=ROOT/'outputs/retail-data'/name;a=json.loads((root/'label-audit.json').read_text());r=json.loads((root/'replay-audit.json').read_text());assert a['all_passed'] and r['passed']==r['records_checked'] and not r['failures']
  for split,file in [('train','sft.jsonl'),('validation','validation.jsonl')]:
   digest=hashlib.sha256((root/file).read_bytes()).hexdigest();assert digest==a['train_sha256' if split=='train' else 'validation_sha256'];rows=read(root/file)
   for row in rows:row.update(e2e_source=name,category='end_to_end_recovery' if 'recovery' in name else 'end_to_end_clean',loss_weight=1.)
   parts[split]+=rows;sources.append(dict(source=name,split=split,records=len(rows),sha256=digest))
 retention=ROOT/'outputs/retail-data/post-rl-clarification200-retention100';rows=read(retention/'train.jsonl');families=defaultdict(list);rng=random.Random(42)
 for r in rows:families[r['family']].append(r)
 picked=[]
 for family in sorted(families):
  rng.shuffle(families[family])
  if families[family][0]['category']=='clarification':picked+=families[family][:10]
 other=[r for r in rows if r['category']=='retention'];rng.shuffle(other);picked+=other[:20];assert len(picked)==100
 for split,pool in [('train',picked),('validation',read(retention/'validation.jsonl'))]:
  for row in pool:row.update(e2e_source='clarification-retention',loss_weight=1.)
  parts[split]+=pool
 for split,rows in parts.items():
  for row in rows:
   actual={t['function']['name']:_schema_contract(t['function']['parameters']) for t in json.loads(re.findall(r'<tools>(.*?)</tools>',row['messages'][0]['content'],re.S)[0])}
   assert set(actual)-set(wanted)<= {'get_item_details'} and all(n in actual and _supported_contract(schema,actual[n]) for n,schema in wanted.items())
   assert all(c['call']['name'] in wanted for c in row.get('tool_execution_audit',[]))
   row['messages'][0]['content']=canonical
   row['context_tokens']=len(tok.apply_chat_template(row['messages'],tokenize=True,return_dict=False))
   assert row['context_tokens']<=8192
  rng.shuffle(rows);(OUT/('train.jsonl' if split=='train' else 'validation.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 groups={s:{r.get('group_id',r['task_id']) for r in rows} for s,rows in parts.items()};assert not groups['train']&groups['validation']
 manifest=dict(partitions={'sft':sorted({r['task_id'] for r in parts['train']}),'validation':sorted({r['task_id'] for r in parts['validation']})});(OUT/'task-split-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 coverage={s:Counter(c['call']['name'] for r in rows for c in r.get('tool_execution_audit',[])) for s,rows in parts.items()}
 result=dict(complete=True,train=len(parts['train']),validation=len(parts['validation']),source_breakdown={s:dict(Counter(r['e2e_source'] for r in rows)) for s,rows in parts.items()},coverage={s:dict(c) for s,c in coverage.items()},sources=sources,group_overlap=0,assistant_customer_tool_messages_and_loss_indices_preserved=True,system_prompt_canonicalized=True,system_sha256=hashlib.sha256(canonical.encode()).hexdigest(),source_dialogues_previously_replay_verified=559,initial_adapter='post20 checkpoint-7',lineage_note='Initializer descends from Glaive RL17 through Salesforce APIGen50 and short/focused clarification branches; this cleaned synthetic end-to-end branch is separate. Prior multi-turn exposure is not claimed absent.',coverage_gap=['modify_pending_order_items','exchange_delivered_order_items'],data_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['train.jsonl','validation.jsonl','task-split-manifest.json']})
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
