"""Grounded ten-example pilot, five requests per existing local Qwen endpoint."""
import concurrent.futures,json,time,re
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer
from tau2_grpo.sft_data import tokenize_assistant_only
OUT=Path('outputs/retail-data/clarification-pilot-10');OUT.mkdir(parents=True,exist_ok=True)
prompt=Path('templates/retail_clarification_2k_generator.txt').read_text()
(OUT/'generator-prompt.txt').write_text(prompt)
rows=[];seen=set()
for line in Path('outputs/retail-data/retail-sft-corrective-v2-final/sft.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['task_id'] in seen or r.get('source_terminal_reward')!=1:continue
 if r['messages'][1]['content'].startswith('Original customer request:'):continue
 for i,m in enumerate(r['messages']):
  if m['role']!='assistant':continue
  match=re.fullmatch(r'<tool_call>(.*?)</tool_call>',m['content'],re.S)
  if not match:continue
  call=json.loads(match[1])
  if call['name'] not in ('find_user_id_by_email','find_user_id_by_name_zip'):continue
  prefix=r['messages'][1:i]
  if any('<tool_response>' in x['content'] for x in prefix):break
  rows.append((r,i,call));seen.add(r['task_id']);break
 if len(rows)==10:break
assert len(rows)==10,len(rows)
write=lambda p,v:p.write_text(json.dumps(v,indent=2))
write(OUT/'source-manifest.json',[{'task_id':r['task_id'],'source_id':r['id'],'source_episode':r.get('source_episode'),'call':c} for r,i,c in rows])

def worker(parity):
 client=OpenAI(base_url=f'http://127.0.0.1:{8005 if parity==0 else 8004}/v1',api_key='local',timeout=180)
 for attempt in range(90):
  try:client.models.list();break
  except Exception:time.sleep(2)
 else:raise RuntimeError('Endpoint not ready')
 results=[]
 for n in range(parity,10,2):
  r,i,call=rows[n]
  variant='ask_then_act' if n<6 else 'ready_to_act' if n<9 else 'partial_answer'
  data={'sample_id':f'clarification-pilot-{n:02d}','split':'train-pilot','scenario_category':'Missing email' if 'email' in call['arguments'] else 'Partial identity','requested_variant':variant,'original_system_policy_and_schemas':r['messages'][0]['content'],'grounded_source_conversation':r['messages'][1:i+1],'required_final_call':call}
  instruction='Create ONE concise rewritten example using this grounded source. Preserve the customer goal. Customer identity values must come only from the source. The FINAL assistant content MUST be the tool-call markup, never prose such as Please hold or I will look you up. End with EXACTLY the required_final_call serialized as <tool_call>{JSON}</tool_call>. No tool observations are needed. For ready_to_act, place all required identity facts in the initial user message. For ask_then_act, have the customer provide missing identity after the question. For partial_answer, split supplied identity facts across answers; if email-only, use a partial/incomplete email followed by the complete source email. Do not invent a second identity. Return the specified JSON object.'
  if variant=='ask_then_act': instruction+=' IMPORTANT: REMOVE all identity names, emails, and ZIP codes from the FIRST user message. Put the exact source identity ONLY in the SECOND user message, after the clarification. First user message retains only the retail request. Do not ask for confirmation of already supplied names.'
  if variant=='partial_answer': instruction+=' IMPORTANT: if the first answer supplies the full name, the second question must ask ONLY for ZIP; never ask for surname again.'
  for retry in range(25):
   cached=OUT/f'{n:02d}-attempt-{retry}.json'
   if cached.exists():
    raw=json.loads(cached.read_text())['response']['choices'][0]['message']['content']
   else:
    response=client.chat.completions.create(model='qwen3.5-4b',messages=[{'role':'system','content':prompt},{'role':'user','content':instruction+'\n'+json.dumps(data)}],temperature=.5,max_tokens=2500,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    raw=response.choices[0].message.content
    (OUT/f'{n:02d}-attempt-{retry}.json').write_text(json.dumps({'request':data,'response':response.model_dump()},indent=2))
   try:
    obj=json.loads(raw);msgs=obj['messages'];assert msgs and msgs[0]['role']=='user'
    assert all(m['role'] in ('user','assistant') for m in msgs)
    assert all(m['role']==('user' if j%2==0 else 'assistant') for j,m in enumerate(msgs))
    match=re.fullmatch(r'<tool_call>(.*?)</tool_call>',msgs[-1]['content'],re.S)
    assert match and json.loads(match[1])==call
    prior=' '.join(m['content'] for m in msgs[:-1] if m['role']=='user')
    assert all(str(v).lower() in prior.lower() for v in call['arguments'].values())
    assert sum('<tool_call>' in m['content'] for m in msgs)==1
    if variant=='ready_to_act':assert len(msgs)==2
    elif variant=='ask_then_act':assert len(msgs)==4
    else:assert len(msgs)==6
    for j,m in enumerate(msgs[:-1]):
     if m['role']!='assistant':continue
     visible=' '.join(x['content'] for x in msgs[:j] if x['role']=='user').lower()
     assert not all(str(v).lower() in visible for v in call['arguments'].values()), 'Unnecessary question: all identity fields already known'
     q=m['content'].lower()
     if 'zip' in call['arguments'] and str(call['arguments']['zip']).lower() not in visible:
      assert not re.search(r'(?:have your zip|zip code from your previous|you (?:already )?(?:provided|gave).{0,25}zip)',q), 'Claims unseen ZIP' 
     request=q[q.rfind('?')-180:] if '?' in q else q
     for field,phrase in [('last_name','last name'),('zip','zip code'),('email','email address')]:
      if field in call['arguments'] and str(call['arguments'][field]).lower() in visible:
       assert not re.search(r'(?:provide|share|what is|give me).*'+phrase,request), 'Repeated known field '+field
    out={'id':data['sample_id'],'domain':'retail','task_id':r['task_id'],'group_id':r['group_id'],'messages':[r['messages'][0]]+msgs,'loss_message_indices':[j+1 for j,m in enumerate(msgs) if m['role']=='assistant'],'variant':variant,'source_id':r['id'],'endpoint':client.base_url.__str__(),'pilot_only':True}
    results.append(out);print('Accepted',n,variant,'task',r['task_id'],flush=True);break
   except Exception as exc:print('Rejected',n,retry,repr(exc),flush=True)
  else:raise RuntimeError(f'Sample {n} failed validation')
 return results
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(worker,p) for p in (0,1)]
 results=[r for f in futures for r in f.result()]
tok=AutoTokenizer.from_pretrained('outputs/hf-export/tiny-aya-global-function-calling-sft-525')
for r in results:
 enc,_=tokenize_assistant_only(tok,r['messages'],8192,r['loss_message_indices']);r['context_tokens']=len(enc['input_ids'])
results.sort(key=lambda r:r['id'])
(OUT/'sft.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
(OUT/'conversations.txt').write_text('\n\n'.join(f"=== {r['id']} | task {r['task_id']} | {r['variant']} ===\n"+'\n\n'.join(m['role'].upper()+': '+m['content'] for m in r['messages'][1:]) for r in results))
write(OUT/'report.json',{'samples':len(results),'endpoint_counts':{str(port):sum(str(port) in r['endpoint'] for r in results) for port in (8004,8005)},'max_context_tokens':max(r['context_tokens'] for r in results),'checks':'alternating roles, variant message count, exact grounded final lookup call, customer-visible arguments, one tool call, production masking and 8192 token limit','limitation':'Grounded in existing source identity lookups; not a live environment replay or coverage of all seven planned categories.'})
print('DONE',OUT,flush=True)
