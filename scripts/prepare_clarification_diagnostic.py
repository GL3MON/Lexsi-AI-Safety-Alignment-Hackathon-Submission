"""Twenty matched missing/supplied pairs, plus twenty fresh held-out identities."""
import json,hashlib,random
from pathlib import Path
from collections import Counter
from tau2_grpo.short_scenario_bank import make_world,tools_from_world,matches
from tau2_grpo.sft_data import tokenize_assistant_only
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('email','first_name','last_name','zip')
def main():
 from tau2.domains.retail.data_model import get_db
 from tau2.domains.retail.utils import RETAIL_TASK_SET_PATH
 from transformers import AutoTokenizer
 base=get_db();catalog={k:v.model_dump(mode='json') for k,v in base.products.items()};tasks={t['id']:t for t in json.loads(RETAIL_TASK_SET_PATH.read_text())};pool=json.loads((ROOT/'outputs/retail-data/split_manifest.json').read_text())['partitions']['rl']
 out=ROOT/'outputs/retail-data/clarification-diagnostic20';out.mkdir(parents=True,exist_ok=True);(out/'worlds').mkdir(exist_ok=True);(out/'examples').mkdir(exist_ok=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'outputs/rl-initialization/post-rl-clarification-sft-best19');system=(ROOT/'templates/retail_apigen_sft_system.txt').read_text();data={};max_tokens=0
 for split,offset in [('train',200),('heldout',300)]:
  rows=[]
  for i in range(20):
   n=offset+i;taskid=pool[i%len(pool)];oid=next(a['arguments']['order_id'] for a in tasks[taskid]['evaluation_criteria']['actions'] if a.get('arguments',{}).get('order_id') in base.orders);world=make_world(base,n,taskid,oid);user=next(iter(world['users'].values()));field=FIELDS[i%4];values=dict(email=user['email'],first_name=user['name']['first_name'],last_name=user['name']['last_name'],zip=user['address']['zip']);label={'email':'email address','first_name':'first name','last_name':'last name','zip':'ZIP code'}
   if field=='email':opening=f'Please find my account using email. My name is {values["first_name"]} {values["last_name"]}.';call=dict(name='find_user_id_by_email',arguments={'email':values['email']})
   else:
    opening='I cannot remember my email. Please find my account using my name and ZIP code.'
    for f in ['first_name','last_name','zip']:
     if f!=field:opening+=f' My {label[f]} is {values[f]}.'
    call=dict(name='find_user_id_by_name_zip',arguments={f:values[f] for f in ['first_name','last_name','zip']})
   tools=tools_from_world(world,catalog);assert getattr(tools,call['name'])(**call['arguments'])==user['user_id'];(out/'worlds'/(world['world_id']+'.json')).write_text(json.dumps(world,indent=2))
   for supplied in [False,True]:
    text=opening+(f' My {label[field]} is {values[field]}.' if supplied else '')
    expected=dict(kind='call',**call) if supplied else dict(kind='ask',field=field)
    target='<tool_call>'+json.dumps(call,separators=(',',':'))+'</tool_call>' if supplied else f'Could you please provide your {label[field]}?'
    prompt=[dict(role='system',content=system),dict(role='user',content=text)];assert matches(expected,target,[call] if supplied else [],text)
    if not supplied:assert values[field] not in text
    row=dict(id=f'diag-{split}-{i:02d}-'+('supplied' if supplied else 'missing'),pair_id=f'{split}-{i:02d}',domain='retail',task_id=world['world_id'],world_id=world['world_id'],field=field,category='call' if supplied else 'ask',expected_action=expected,messages=prompt+[dict(role='assistant',content=target)],loss_message_indices=[2],loss_weight=1.)
    encoded,spans=tokenize_assistant_only(tok,row['messages'],8192,[2]);assert len(spans)==1;a,b=spans[0];assert all(v==-100 for v in encoded['labels'][:a]);assert encoded['labels'][a:b]==encoded['input_ids'][a:b];max_tokens=max(max_tokens,len(encoded['input_ids']));rows.append(row)
  random.Random(42).shuffle(rows);data[split]=rows;(out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  for f in FIELDS:
   pair=[r for r in rows if r['field']==f][:1];pid=pair[0]['pair_id'];pair=sorted([r for r in rows if r['pair_id']==pid],key=lambda r:r['category'])
   (out/'examples'/f'{split}-{f}.txt').write_text('\n\n'.join(r['category'].upper()+':\n'+r['messages'][1]['content']+'\n\nTARGET:\n'+r['messages'][2]['content'] for r in pair))
 assert not {r['world_id'] for r in data['train']}&{r['world_id'] for r in data['heldout']}
 manifest=dict(complete=True,training_pairs=20,training_examples=40,heldout_pairs=20,heldout_examples=40,fields=list(FIELDS),per_split=dict(ask=20,call=20),max_tokens=max_tokens,world_overlap=0,prompt='Exact saved retail SFT system and Jinja; only the one supplied field differs within a matched pair.',labels='One next assistant decision; prompt/user masked; equal per-example weights; no later customer/tool response in inputs.',source='Fresh synthetic worlds200–219 and300–319 constructed from original RL structural seeds; identities and states have not appeared in previous generated banks.',initializer='SFT19',purpose='Memorization and transfer diagnostic, not a benchmark or production checkpoint',data_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in data})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
