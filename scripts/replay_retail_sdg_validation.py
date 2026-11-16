"""Rebuild each SDG world and re-execute saved tool calls against real RetailTools."""
import argparse,ast,copy,concurrent.futures,inspect,json,textwrap,traceback
from pathlib import Path
from types import SimpleNamespace
SEED=None;BUILD=None;GEN=None

def initialize():
 global SEED,BUILD,GEN
 import generate_retail_replay_sft as gen
 from tau2.domains.retail.data_model import get_db
 SEED=get_db();GEN=gen
 # Reuse the trusted deterministic scenario-construction prefix, stopping
 # before any teacher API request. No generated text is compiled or executed.
 fn=ast.parse(textwrap.dedent(inspect.getsource(gen.Generator.generate))).body[0]
 prefix=[]
 for node in fn.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='customer_system' for t in node.targets):break
  prefix.append(node)
 fn.name='_audit_build_world';fn.body=prefix+[ast.parse('return tools, goals, destination, schemas, payment').body[0]]
 module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(vars(gen));exec(compile(module,'<trusted-sdg-world-builder>','exec'),namespace);BUILD=namespace[fn.name]

def check(path):
 try:
  row=json.loads(Path(path).read_text());n=int(row['id'].rsplit('-',1)[-1]);delta=row['seed']-20261113-n*101;assert delta>=0 and delta%100003==0;attempt=delta//100003
  calc=next((g['kind'] for g in row['goals'] if g['action']=='calculate'),None);kind=row['masked_faults'][0]['kind'] if row['masked_faults'] else None
  actor=SimpleNamespace(a=SimpleNamespace(seed=20261113,family_prefix='retail-sdg-v2',corruption_kind=kind,calculator_scenario=calc),seed=SEED,run_tag='audit')
  tools,goals,destination,schemas,payment=BUILD(actor,n,attempt,row['mode']=='recovery')
  assert goals==row['goals'],'rebuilt scenario goals differ'
  faults={f['message_index'] for f in row['masked_faults']};pending=None;audit=[];tool_count=0;auth=False
  for i,message in enumerate(row['messages']):
   if message['role']=='assistant':
    try:calls=GEN.parse_calls(message['content'])
    except ValueError:
     assert i in faults;calls=[None]
    if not calls:continue
    assert len(calls)==1 and pending is None
    if i in faults:
     clone=copy.deepcopy(tools);before=clone.db.model_dump(mode='json');call,body,ok=GEN.execute(clone,schemas,message['content']);assert not ok and clone.db.model_dump(mode='json')==before
    else:
     call0=calls[0]
     if not call0['name'].startswith('find_'):assert auth,'tool before authentication'
     if call0['name']=='calculate':
      cg=next((g for g in goals if g['action']=='calculate'),None)
      visible='\n'.join(m['content'] for m in row['messages'][:i] if m['role'] in ('user','tool'))
      GEN.validate_calculation(call0['arguments']['expression'],audit,cg.get('budget') if cg else None,visible)
     call,body,ok=GEN.execute(tools,schemas,message['content']);assert ok,'unmasked tool failure'
     if call['name'].startswith('find_'):auth=True
     audit.append({'call':call,'result':body})
    pending={'name':call['name'] if call else '__runtime__','content':body};tool_count+=1
   elif message['role']=='tool':
    assert pending is not None,'orphan tool observation'
    recorded=json.loads(message['content']);assert pending==recorded,('tool observation differs',pending,recorded)
    pending=None
  assert pending is None
  assert GEN.Generator.goals_met(goals,tools,destination,audit),'final world does not satisfy goals'
  for g in goals:
   if g['action']=='calculate':assert GEN.answer_matches(g,row['messages'][-1]['content'])
  return {'id':row['id'],'passed':True,'tool_calls_replayed':tool_count}
 except Exception as e:return {'id':Path(path).stem,'passed':False,'error':str(e)[:1400],'traceback':traceback.format_exc()[-1400:]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/retail-data/retail-sdg-v2-700');p.add_argument('--output',default='outputs/diagnostics/retail-sdg-v2-final-audit/replay.json');p.add_argument('--workers',type=int,default=4);a=p.parse_args();files=sorted((Path(a.source)/'records').glob('*.json'));results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize) as pool:
  for i,result in enumerate(pool.map(check,map(str,files),chunksize=8),1):
   results.append(result)
   if i%50==0:print('Replayed',i,'/',len(files),flush=True)
 report={'records_checked':len(results),'passed':sum(r['passed'] for r in results),'tool_calls_replayed':sum(r.get('tool_calls_replayed',0) for r in results),'failures':[r for r in results if not r['passed']]}
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 if report['failures']:raise SystemExit(1)
if __name__=='__main__':main()
