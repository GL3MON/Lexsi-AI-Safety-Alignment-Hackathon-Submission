"""Fresh 200 recovery + 500 clean retail traces, concurrently on MIG0 and MIG2."""
import json,subprocess,sys,time,urllib.request,hashlib
from pathlib import Path
from tau2_grpo.rl_rollout import atomic_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/retail-data/retail-sdg-v2-700'
SAFE=('id','task_id','domain','group_id','split','messages','loss_message_indices','context_tokens','mode','chat_template_path')
JOBS=[('recovery-json',8005,40000,40,0,'json_structure',None),('recovery-tool',8004,50000,40,0,'unknown_tool',None),('recovery-args',8005,60000,40,0,'argument_schema',None),('recovery-id',8004,70000,40,0,'wrong_identifier',None),('recovery-operation',8005,80000,40,0,'wrong_operation',None),('clean-a',8004,90000,0,80,None,None),('clean-b',8005,100000,0,80,None,None),('clean-c',8004,110000,0,80,None,None),('clean-d',8005,120000,0,80,None,None),('clean-e',8004,130000,0,80,None,None),('calculator-a',8005,140000,0,50,None,'mixed'),('calculator-b',8004,150000,0,50,None,'mixed')]
CACHE={};last_count=-1

def merge(stage):
 global last_count
 for name,*_ in JOBS:
  for p in (OUT/'workers'/name/'records').glob('*.json'):
   if p.name not in CACHE:
    row=json.loads(p.read_text());assert row['role_review']['role_consistent'] and row['semantic_review']['accept'] and row['context_tokens']<=8192
    assert (row['mode']=='clean' and not row['masked_faults']) or (row['mode']=='recovery' and len(row['masked_faults'])==1)
    CACHE[p.name]=row;atomic_json(OUT/'records'/p.name,row)
    txt=p.with_suffix('.txt')
    if txt.exists():(OUT/'records'/txt.name).write_text(txt.read_text())
 rows=sorted(CACHE.values(),key=lambda r:r['id']);assert len({r['id'] for r in rows})==len(rows)
 if len(rows)!=last_count:
  for filename,split in [('sft','train'),('validation','validation'),('all',None)]:
   dest=OUT/(filename+'.jsonl');tmp=dest.with_suffix('.tmp');tmp.write_text(''.join(json.dumps({k:r[k] for k in SAFE},ensure_ascii=False)+'\n' for r in rows if split is None or r['split']==split));tmp.replace(dest)
  atomic_json(OUT/'task-split-manifest.json',{'partitions':{'sft':[r['task_id'] for r in rows if r['split']=='train'],'validation':[r['task_id'] for r in rows if r['split']=='validation']},'split_unit':'scenario_family'})
  last_count=len(rows)
 types={kind:sum(r['mode']=='recovery' and r['masked_faults'][0]['kind']==kind for r in rows) for kind in ['json_structure','unknown_tool','argument_schema','wrong_identifier','wrong_operation']}
 report={'stage':stage,'completed':len(rows),'target':700,'recovery_completed':sum(r['mode']=='recovery' for r in rows),'recovery_target':200,'clean_completed':sum(r['mode']=='clean' for r in rows),'clean_target':500,'calculator_completed':sum(any(g['action']=='calculate' for g in r['goals']) for r in rows),'calculator_target':100,'calculator_scenarios':{kind:sum(any(g['action']=='calculate' and g['kind']==kind for g in r['goals']) for r in rows) for kind in ['refund_total','variant_difference','budget']},'recovery_types':types,'max_context_tokens':max((r['context_tokens'] for r in rows),default=0),'updated_at':time.time()}
 atomic_json(OUT/'progress.json',report);return report

def healthy(port):
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:return r.status==200
 except Exception:return False

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'records').mkdir(exist_ok=True)
 for name,*_ in JOBS:
  for d in ['records','attempts','rejected','skipped']:(OUT/'workers'/name/d).mkdir(parents=True,exist_ok=True)
 config={'seed':20261113,'jobs':JOBS,'target_recovery':200,'target_clean':500,'concurrent_clean_and_recovery':True,'per_scenario_attempts':4,'per_turn_draft_attempts':3,'role_gate':'deterministic each turn plus focused Qwen audit before acceptance','semantic_and_real_db_checks':True,'chat_template':'templates/retail_sdg_v2.jinja','template_sha256':hashlib.sha256((ROOT/'templates/retail_sdg_v2.jinja').read_bytes()).hexdigest(),'old_samples_reused':False,'calculator_scenario_target':100}
 atomic_json(OUT/'config.json',config)
 running={};done=set();restarts={};started=time.time()
 try:
  while len(done)<len(JOBS):
   for name,port,offset,rec,clean,kind,calculator in JOBS:
    if name in done:continue
    if name not in running:
     if not healthy(port):
      if time.time()-started>900:raise RuntimeError(f'Endpoint {port} unavailable')
      continue
     wd=OUT/'workers'/name
     command=[sys.executable,'-u','scripts/generate_retail_replay_sft.py','--output',str(wd),'--endpoint',f'http://127.0.0.1:{port}/v1','--start-index',str(offset),'--recovery-count',str(rec),'--clean-count',str(clean),'--seed','20261113','--family-prefix','retail-sdg-v2','--balanced-validation','--max-attempts','4']
     if kind:command+=['--corruption-kind',kind]
     if calculator:command+=['--calculator-scenario',calculator]
     with (wd/'generation.log').open('a') as log:child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     running[name]=child;atomic_json(wd/'process.json',{'pid':child.pid,'command':command,'started_at':time.time()})
    result=running[name].poll()
    if result is not None:
     if result:
      restarts[name]=restarts.get(name,0)+1
      if restarts[name]>3:raise RuntimeError(f'{name} failed repeatedly; see its log')
      del running[name]
     else:done.add(name)
   report=merge('generating');atomic_json(OUT/'pipeline-status.json',{**report,'jobs':{name:{'pid':p.pid,'finished':name in done,'restarts':restarts.get(name,0)} for name,p in running.items()}})
   if len(done)<len(JOBS):time.sleep(5)
  report=merge('validating');assert report['recovery_completed']==200 and report['clean_completed']==500 and all(n==40 for n in report['recovery_types'].values())
  from tau2_grpo.sft_data import read_records,validate_task_partitions
  train=read_records(OUT/'sft.jsonl');val=read_records(OUT/'validation.jsonl');validate_task_partitions(train,val,json.loads((OUT/'task-split-manifest.json').read_text()))
  assert len(train)==630 and len(val)==70
  assert report['calculator_completed']==100 and all(n>0 for n in report['calculator_scenarios'].values())
  report=merge('complete');atomic_json(OUT/'pipeline-status.json',{**report,'train':630,'validation':70,'role_gate_passed_for_every_record':True,'training_started':False})
 except Exception as e:
  atomic_json(OUT/'pipeline-status.json',{'stage':'failed','error':str(e),'updated_at':time.time()});raise
if __name__=='__main__':main()
