"""Focused Qwen audit for customer/agent perspective inversion in synthetic traces."""
import argparse,concurrent.futures,json,re,time
from pathlib import Path
from openai import OpenAI
from tau2_grpo.rl_rollout import atomic_json
SCHEMA={'type':'object','properties':{'role_consistent':{'type':'boolean'},'flags':{'type':'array','items':{'type':'object','properties':{'message_index':{'type':'integer'},'role':{'type':'string','enum':['assistant','user']},'evidence':{'type':'string'},'reason':{'type':'string'}},'required':['message_index','role','evidence','reason'],'additionalProperties':False}}},'required':['role_consistent','flags'],'additionalProperties':False}
SYSTEM='''Audit dialogue SPEAKER ROLES only, not tool correctness. The assistant is a retail support AGENT; user is the CUSTOMER. Flag assistant text that speaks as the customer about their own purchases, asks the agent to fix their own order, or copies a customer's first-person request as its own. Flag user text that impersonates the support agent, claims to have looked up the customer's account, asks the customer for authentication, or promises to process the customer's request. An assistant describing what it WILL do for YOUR order is normal. An assistant quoting or explicitly attributing the customer's words is normal. A customer saying I ordered, I want a return, or please proceed is normal. An assistant saying I don't remember my order number; I ordered headphones; please process my return is WRONG. A user saying I've located your recent orders; would you like me to process your refund is WRONG. Use the entire dialogue for context. Return role_consistent=true and flags=[] if no clear perspective inversion exists. For a flag copy a short EXACT evidence substring from that message. Do not flag ordinary acknowledgments, thank-you replies, product questions or requests for confirmation.'''
def audit(row,out,endpoint):
 target=out/'records'/(row['id']+'.json')
 if target.exists():return json.loads(target.read_text())
 dialogue=[]
 for i,m in enumerate(row['messages']):
  if m['role'] not in ('assistant','user'):continue
  if m['role']=='user' and m['content'].startswith('<tool_response>'):continue
  text=re.sub(r'<tool_call>.*?</tool_call>','[tool call]',m['content'],flags=re.S)
  dialogue.append({'message_index':i,'role':m['role'],'text':text})
 with OpenAI(base_url=endpoint,api_key='local',timeout=180,max_retries=2) as c:
  response=c.chat.completions.create(model='qwen3.5-4b',messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(dialogue,ensure_ascii=False)}],temperature=0,max_tokens=900,response_format={'type':'json_schema','json_schema':{'name':'role_audit','schema':SCHEMA,'strict':True}},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
 atomic_json(out/'requests'/(row['id']+'.json'),{'dialogue':dialogue,'response':response.model_dump()})
 if response.choices[0].finish_reason=='length':raise ValueError('truncated role audit '+row['id'])
 result=json.loads(response.choices[0].message.content)
 for flag in result['flags']:
  i=flag['message_index'];assert 0<=i<len(row['messages']) and row['messages'][i]['role']==flag['role']
  assert flag['evidence'] and flag['evidence'] in row['messages'][i]['content'],('unverifiable evidence',row['id'],flag)
 assert result['role_consistent']==(not result['flags'])
 result.update(id=row['id'],mode=row['mode'],split=row['split'],supervised_flag_indices=[f['message_index'] for f in result['flags'] if f['message_index'] in row['loss_message_indices']])
 atomic_json(target,result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/retail-data/retail-replay-2000');p.add_argument('--output',default='outputs/diagnostics/retail-replay-sft-audit/role-review');p.add_argument('--endpoint',default='http://127.0.0.1:8004/v1');p.add_argument('--workers',type=int,default=2);p.add_argument('--ids',nargs='*');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for d in ['records','requests']:(out/d).mkdir(exist_ok=True)
 rows=[json.loads(p.read_text()) for p in sorted((Path(a.source)/'records').glob('*.json'))]
 if a.ids:rows=[r for r in rows if r['id'] in a.ids]
 # Verify the known role swap early, then evaluate the rest.
 rows.sort(key=lambda r:(r['id']!='retail-replay-20316',r['id']))
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(audit,r,out,a.endpoint):r['id'] for r in rows}
  for future in concurrent.futures.as_completed(futures):
   try:results.append(future.result())
   except Exception as e:
    atomic_json(out/'errors'/f'{futures[future]}.json',{'id':futures[future],'error':str(e)})
    print('AUDIT ERROR',futures[future],str(e),flush=True)
   report={'completed':len(results),'target':len(rows),'flagged':sum(not r['role_consistent'] for r in results),'flagged_ids':[r['id'] for r in results if not r['role_consistent']],'updated_at':time.time()}
   atomic_json(out/'progress.json',report)
   if len(results)%25==0:print(json.dumps(report),flush=True)
 if len(results)!=len(rows):raise RuntimeError('Some role audits failed; inspect errors and resume')
 atomic_json(out/'summary.json',{'completed':len(results),'flagged':[r for r in results if not r['role_consistent']]});print('DONE',len(results),'flagged',sum(not r['role_consistent'] for r in results),flush=True)
if __name__=='__main__':main()
