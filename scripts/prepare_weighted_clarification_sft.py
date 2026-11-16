"""Split verified conversations into ask decisions and supplied-answer call controls."""
import copy,json,hashlib,random
from pathlib import Path
from collections import Counter
from tau2_grpo.sft_data import tokenize_assistant_only
ROOT=Path(__file__).resolve().parents[1]
MASS={'clarification':.6,'followup_call':.2,'retention':.2}
def main():
 source=ROOT/'outputs/retail-data/post-rl-clarification200-retention100';out=ROOT/'outputs/retail-data/weighted-clarification-decisions';out.mkdir(parents=True,exist_ok=True);(out/'examples').mkdir(exist_ok=True)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19')
 datasets={};audits={}
 for split in ['train','validation']:
  rows=[]
  for original in map(json.loads,(source/(split+'.jsonl')).read_text().splitlines()):
   if original['category']=='clarification':
    first,*later=original['loss_message_indices'];assert later and '<tool_call>' not in original['messages'][first]['content']
    ask=copy.deepcopy(original);ask.update(id=original['id']+'-ask',messages=original['messages'][:first+1],loss_message_indices=[first],category='clarification',decision='required_input_missing');rows.append(ask)
    follow=copy.deepcopy(original);follow.update(id=original['id']+'-call',loss_message_indices=later,category='followup_call',decision='required_input_received');rows.append(follow)
   else:
    r=copy.deepcopy(original);r.update(decision='retain_verified_skill');rows.append(r)
  counts=Counter(r['category'] for r in rows);audit=[]
  for i,row in enumerate(rows):
   row['source_trajectory_verification']=row.pop('verification')
   row['verification']=dict(source_trajectory_replayed=True,decision_targets_verified=True,full_goal_completed_in_record=row['category']!='clarification')
   row['loss_weight']=MASS[row['category']]/(counts[row['category']]/len(rows))
   encoded,spans=tokenize_assistant_only(tok,row['messages'],8192,row['loss_message_indices']);selected=set(row['loss_message_indices']);covered=set();assistant_indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
   for index,(a,b) in zip(assistant_indices,spans,strict=True):
    if index in selected:
     assert encoded['labels'][a:b]==encoded['input_ids'][a:b];covered.update(range(a,b))
    else:assert all(t==-100 for t in encoded['labels'][a:b])
   assert all(v==-100 for j,v in enumerate(encoded['labels']) if j not in covered)
   if row['category']=='clarification':
    assert len(selected)==1 and not any('<tool_call>' in row['messages'][j]['content'] for j in selected)
    assert row['messages'][-1]['role']=='assistant','No future customer reply may enter ask-decision input'
   elif row['category']=='followup_call':assert all('<tool_call>' in row['messages'][j]['content'] for j in selected)
   audit.append(dict(id=row['id'],category=row['category'],loss_weight=row['loss_weight'],tokens=len(encoded['input_ids']),supervised_tokens=len(covered),targets=[row['messages'][j]['content'] for j in sorted(selected)]))
   if (i+1)%50==0:print('AUDITED',split,i+1,len(rows),flush=True)
  random.Random(42).shuffle(rows);datasets[split]=rows;audits[split]=audit
  (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  for category in MASS:
   r=next(r for r in rows if r['category']==category)
   (out/'examples'/f'{split}-{category}.txt').write_text('\n\n'.join(m['role'].upper()+(' [TARGET]' if i in r['loss_message_indices'] else ' [MASKED]')+':\n'+m['content'] for i,m in enumerate(r['messages'])))
  expected={c:sum(r['loss_weight'] for r in rows if r['category']==c)/len(rows) for c in MASS}
  assert all(abs(expected[c]-MASS[c])<1e-10 for c in MASS)
 assert len(datasets['train'])==500 and len(datasets['validation'])==68
 assert not {r['world_id'] for r in datasets['train']}&{r['world_id'] for r in datasets['validation']}
 for split,rows in datasets.items():assert len({json.dumps(r['messages'],sort_keys=True) for r in rows})==len(rows)
 manifest=dict(complete=True,initial_checkpoint='selected clarification-retention SFT19',initial_adapter_sha256='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811',records={s:len(r) for s,r in datasets.items()},counts={s:dict(Counter(r['category'] for r in rs)) for s,rs in datasets.items()},objective='Mean each example over its supervised tokens, multiply by fixed category weight, average examples in batch; ordinary gradient accumulation averages microbatch losses.',objective_mass=MASS,max_tokens=max(a['tokens'] for rows in audits.values() for a in rows),source='Split previously tool-executed and state-hash-verified trajectories into missing-input ask decisions and subsequent supplied-answer call controls. Historical assistant turns remain masked.',data_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in datasets},train_validation_world_overlap=0,evaluation_worlds=[80,85,86],evaluation_worlds_used_in_sft=False)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'mask-audit.json').write_text(json.dumps(audits,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
