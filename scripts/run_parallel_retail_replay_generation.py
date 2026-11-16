"""MIG0+MIG2 generation: 500 accepted recovery samples, then 1500 clean."""
import json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
from tau2_grpo.rl_rollout import atomic_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/retail-data/retail-replay-2000'
SAFE=('id','task_id','domain','group_id','split','messages','loss_message_indices','context_tokens','mode','chat_template_path')
WORKERS=[('mig2',8004,0,250),('mig2-lane1',8004,10000,0),('mig0',8005,20000,250),('mig0-lane1',8005,30000,0)]
def state(stage,**kw):atomic_json(OUT/'pipeline-status.json',{'stage':stage,'updated_at':time.time(),**kw})
def merge(stage):
    rows=[];skipped=[]
    for name,_,_,_ in WORKERS:
        wd=OUT/'workers'/name
        for p in sorted((wd/'records').glob('*.json')):
            row=json.loads(p.read_text());rows.append(row)
            dest=OUT/'records'/p.name
            if not dest.exists():
                atomic_json(dest,row)
                if p.with_suffix('.txt').exists():shutil.copy2(p.with_suffix('.txt'),dest.with_suffix('.txt'))
        skipped += [{'worker':name,**json.loads(p.read_text())} for p in (wd/'skipped').glob('*.json')]
    rows.sort(key=lambda r:r['id'])
    assert len({r['id'] for r in rows})==len(rows)
    for filename,split in [('sft','train'),('validation','validation'),('all',None)]:
        path=OUT/(filename+'.jsonl');temp=path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps({k:r[k] for k in SAFE},ensure_ascii=False)+'\n' for r in rows if split is None or r['split']==split));temp.replace(path)
    atomic_json(OUT/'task-split-manifest.json',{'partitions':{'sft':[r['task_id'] for r in rows if r['split']=='train'],'validation':[r['task_id'] for r in rows if r['split']=='validation']},'split_unit':'scenario_family'})
    atomic_json(OUT/'skipped-scenarios.json',skipped)
    report={'stage':stage,'completed':len(rows),'recovery_completed':sum(r['mode']=='recovery' for r in rows),'clean_completed':sum(r['mode']=='clean' for r in rows),'target':2000,'max_context_tokens':max((r['context_tokens'] for r in rows),default=0),'skipped_scenarios':len(skipped),'updated_at':time.time()}
    atomic_json(OUT/'progress.json',report)
    return report

def ready(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:return response.status==200
    except Exception:return False

def phase(stage,clean):
    running={};finished=set();started=time.time()
    while len(finished)<len(WORKERS):
        for name,port,offset,recovery_target in WORKERS:
            if name in finished:continue
            if clean==0 and recovery_target==0:
                finished.add(name);continue
            if name not in running:
                if not ready(port):
                    if time.time()-started>900:raise RuntimeError(f'{name} endpoint not healthy after 15 minutes')
                    continue
                wd=OUT/'workers'/name;log=(wd/'generation.log').open('a')
                command=[sys.executable,'-u','scripts/generate_retail_replay_sft.py','--output',str(wd),'--endpoint',f'http://127.0.0.1:{port}/v1','--recovery-count',str(recovery_target),'--clean-count',str(clean),'--start-index',str(offset)]
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);log.close();running[name]=p
                atomic_json(wd/'process.json',{'pid':p.pid,'command':command,'phase':stage})
            result=running[name].poll()
            if result is not None:
                if result:raise RuntimeError(f'{name} worker exited {result}; see its generation.log')
                finished.add(name)
        report=merge(stage);state(stage,**{k:v for k,v in report.items() if k!='stage'},workers={name:{'pid':p.pid,'finished':name in finished} for name,p in running.items()})
        if len(finished)<len(WORKERS):time.sleep(10)

def main():
    for name,_,_,_ in WORKERS:
        for d in ['records','attempts','rejected','skipped']:(OUT/'workers'/name/d).mkdir(parents=True,exist_ok=True)
    # Restore only records not already owned by a worker; each lane owns a disjoint ID range.
    owned={p.name for name,_,_,_ in WORKERS for p in (OUT/'workers'/name/'records').glob('*.json')}
    for p in (OUT/'records').glob('*.json'):
        if p.name in owned:continue
        owner=next(name for name,_,offset,_ in WORKERS if offset<=int(p.stem)<offset+10000)
        dest=OUT/'workers'/owner/'records'/p.name
        if not dest.exists():
            shutil.copy2(p,dest)
            if p.with_suffix('.txt').exists():shutil.copy2(p.with_suffix('.txt'),dest.with_suffix('.txt'))
    skipped=OUT/'workers/mig2/skipped/0014.json'
    if not skipped.exists():atomic_json(skipped,{'scenario':14,'mode':'recovery','attempts':100,'reason':'previous_run_attempts_exhausted','skipped_at':time.time()})
    try:
        phase('recovery',0)
        assert merge('recovery')['recovery_completed']==500
        phase('clean',375)
        report=merge('complete');assert report['recovery_completed']==500 and report['clean_completed']==1500
        from tau2_grpo.sft_data import read_records,validate_task_partitions
        train=read_records(OUT/'sft.jsonl');val=read_records(OUT/'validation.jsonl')
        validate_task_partitions(train,val,json.loads((OUT/'task-split-manifest.json').read_text()))
        state('complete',**{k:v for k,v in report.items() if k!='stage'},training_launched=False)
    except Exception as e:
        state('failed',error=str(e));raise
if __name__=='__main__':main()
