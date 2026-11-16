#!/usr/bin/env python3
"""Local live rollout browser. Optional read-only synchronization from private HF buckets."""
import argparse
import json
import logging
import re
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('rollout-viewer')


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def confined(root, relative):
    candidate=(root/relative).resolve()
    if not candidate.is_relative_to(root.resolve()): raise ValueError('Invalid path')
    return candidate


def episode_outcome(data, reward):
    if not data.get('complete',reward is not None):return 'pending'
    full=data.get('short_task_success',(data.get('reward_details') or {}).get('short_success'))
    if data.get('native_reward') is not None:full=data['native_reward']==1
    if full is True or (full is None and reward==1):return 'success'
    details=data.get('reward_details') or {}
    if details.get('verified_steps',0)>0 or details.get('progress',0)>0 or details.get('tau2_sum',0)>0 or details.get('unique_correct_calls',0)>0:return 'partial'
    if isinstance(reward,(int,float)) and 0<reward<1:return 'partial'
    return 'failure'


class Store:
    def __init__(self, jobs):
        self.jobs={j['job_id']:j for j in jobs}
        self.sync_state={}
        self.stamps={}
        self.lock=threading.Lock()

    def folder(self, job_id):
        if job_id not in self.jobs: raise ValueError('Unknown job')
        return ROOT/'outputs/hf-jobs'/job_id

    def sync(self, seconds):
        from huggingface_hub import HfApi
        api=HfApi()
        while True:
            # Keep old completed runs available from their local cache. Mirror the newest job.
            job=next(iter(self.jobs.values()));jid=job['job_id'];folder=self.folder(jid)
            try:
                status=read_json(folder/'status.json',{})
                if time.time()-status.get('checked_at',0)>30:
                    current=api.inspect_job(job_id=jid,namespace=job.get('account','GL3MON'))
                    folder.mkdir(parents=True,exist_ok=True)
                    updated=dict(job_id=jid,status=current.status.stage,message=current.status.message,checked_at=time.time())
                    temporary=folder/'status.viewer.tmp';temporary.write_text(json.dumps(updated));temporary.replace(folder/'status.json')
                downloads=[];stamps={}
                for entry in api.list_bucket_tree(job['checkpoint_bucket'],prefix=job['bucket_prefix'],recursive=True):
                    name=entry.path;leaf=name.rsplit('/',1)[-1]
                    wanted=leaf in ('groups.json','summary.json','result.json','error.txt','progress.json','selection.json') or bool(re.fullmatch(r'episode-\d+\.json',leaf)) or ('/short-' in name and name.endswith('.json')) or ('/tau2' in name and bool(re.fullmatch(r'task-\d+\.json',leaf)))
                    if not wanted or '/checkpoint-' in name:continue
                    if not name.startswith(job['bucket_prefix']+'/'):continue
                    stamp=getattr(entry,'xet_hash',None) or str(getattr(entry,'mtime',''))
                    dest=confined(folder/'bucket',name)
                    if self.stamps.get((jid,name))==stamp and dest.exists():continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    # Download to side files; never expose a half-written JSON document.
                    downloads.append((name,dest.with_suffix(dest.suffix+'.download')))
                    stamps[name]=(stamp,dest)
                if downloads:
                    api.download_bucket_files(job['checkpoint_bucket'],downloads,raise_on_missing_files=False)
                    for name,temp in downloads:
                        if temp.exists():
                            stamp,dest=stamps[name];temp.replace(dest);self.stamps[(jid,name)]=stamp
                with self.lock:self.sync_state[jid]=dict(last_success=time.time(),error=None,files_updated=len(downloads))
            except Exception as error:
                LOG.warning('Bucket sync: %s',error)
                with self.lock:
                    state=self.sync_state.setdefault(jid,{})
                    state.update(error=f'{type(error).__name__}: {error}')
            time.sleep(seconds)

    def sources(self):
        return [dict(id=jid,name=j.get('task',jid),url=j.get('url'),status=read_json(self.folder(jid)/'status.json',{}).get('status','UNKNOWN')) for jid,j in self.jobs.items()]

    def index(self,jid):
        root=self.folder(jid);bucket=root/'bucket';groups=[];rows=[];evaluation=[]
        for f in sorted(bucket.rglob('groups.json')):
            data=read_json(f,[])
            if isinstance(data,list):groups.extend(data)
        by_group={g['group']:g for g in groups}
        # Include saved text-only historical rollouts. Prefer richer JSON when mirrored.
        paths={f.with_suffix('') for f in bucket.rglob('*') if f.is_file() and f.suffix in ('.json','.txt') and (re.fullmatch(r'episode-\d+',f.stem) or (f.parent.name.startswith('tau2') and re.fullmatch(r'task-\d+',f.stem)) or (f.parent.name.startswith('short-') and (f.stem.endswith('-bank') or re.fullmatch(r'\d+-.+',f.stem)))) and f.name not in ('summary.json','error.txt')}
        for stem in sorted(paths):
            group_match=re.fullmatch(r'group-(\d+)',stem.parent.name)
            group=int(group_match[1]) if group_match else None
            phase=(stem.parent.parent.name+'/'+stem.parent.name if self.jobs[jid].get('task')=='post20-paired-eval' else stem.parent.name) if group is None else ('smoke' if 'smoke' in stem.parts else 'training')
            num=re.fullmatch(r'episode-(\d+)',stem.name);rollout=int(num[1]) if num else None
            data=read_json(stem.with_suffix('.json'),{}) or {}
            g=by_group.get(group,{}) if phase=='training' else {}
            reward=data.get('reward')
            if reward is None and rollout is not None and rollout<len(g.get('rewards',[])):reward=g['rewards'][rollout]
            advantage=None
            if rollout is not None:
                if g.get('advantages') and rollout<len(g['advantages']):advantage=g['advantages'][rollout]
                elif g.get('skipped_tied_group'):advantage=0.
            task=data.get('task_id',g.get('task_id',stem.name))
            details=data.get('reward_details') or {}
            rows.append(dict(assistance_level=(data.get('assistance') or {}).get('level','none'),prefix_actions=(data.get('assistance') or {}).get('prefix_actions',0),outcome=episode_outcome(data,reward),verified_steps=details.get('verified_steps'),required_steps=details.get('required_steps'),key=stem.relative_to(bucket).as_posix(),phase=phase,group=group,rollout=rollout,task_id=task,reward=reward,advantage=advantage,skipped=bool(g.get('skipped_tied_group')),complete=data.get('complete',reward is not None),error=data.get('error'),turns=len(data['actions']) if 'actions' in data else None,policy_version=data.get('policy_version',g.get('policy_version'))))
        for f in sorted(bucket.rglob('summary.json')):
            data=read_json(f,{})
            if 'results' in data:evaluation.append(dict(phase=f.parent.name,passed=data.get('passed'),total=data.get('total'),results=data['results'],pairs_both_correct=data.get('pairs_both_correct'),pairs_complete=data.get('pairs_complete')))
        if self.jobs[jid].get('evaluation_invalid'):evaluation=[];rows=[]
        rewards=[r for g in groups for r in g.get('rewards',[]) if isinstance(r,(int,float))]
        stats=dict(groups=len(groups),rollouts=len(rewards),successes=sum(r==1 for r in rewards),failures=sum(r==-1 for r in rewards),zeros=sum(r==0 for r in rewards),updates=max((g.get('optimizer_steps_total',0) for g in groups),default=0),tied_groups=sum(bool(g.get('skipped_tied_group')) for g in groups),mixed_groups=sum(len(set(g.get('rewards',[])))>1 for g in groups),mean_reward=sum(rewards)/len(rewards) if rewards else None)
        if self.jobs[jid].get('real_tau2_rewards'):
            native=[r for g in groups for r in g.get('native_rewards',[]) if isinstance(r,(int,float))]
            stats['successes']=sum(r==1 for r in native)
        with self.lock:sync=dict(self.sync_state.get(jid,{}))
        dpo={}
        if self.jobs[jid].get('task')=='clarification-dpo':
            progress=next(iter(sorted(bucket.rglob('progress.json'))),None)
            selection=next(iter(sorted(bucket.rglob('selection.json'))),None)
            dpo=dict(progress=read_json(progress,{}) if progress else {},selection=read_json(selection,{}) if selection else {})
        status=read_json(root/'status.json',{})
        failure=bucket/self.jobs[jid].get('bucket_prefix','__no_remote_prefix__')/'error.txt'
        if failure.exists():status=dict(status,status='FAILED',message=failure.read_text()[-1200:])
        return dict(job=self.jobs[jid],status=status,sync=sync,groups=groups,episodes=rows,evaluation=evaluation,stats=stats,dpo=dpo,server_time=time.time())

    def episode(self,jid,key,raw=False):
        root=self.folder(jid)/'bucket';stem=confined(root,key)
        if stem.suffix:raise ValueError('Expected episode key')
        data=read_json(stem.with_suffix('.json'))
        if data is None:
            txt=stem.with_suffix('.txt')
            if not txt.is_file():raise FileNotFoundError('Rollout has not synchronized yet')
            text=txt.read_text();messages=[]
            for part in re.split(r'\n\n(?=(?:SYSTEM|USER|ASSISTANT|TOOL):\n)',text):
                role,sep,content=part.partition(':\n')
                messages.append(dict(role=role.lower() if sep else 'transcript',content=content if sep else part))
            return dict(messages=messages,actions=[],text_only=True)
        if raw:return data
        # The browser does not need multi-thousand-token IDs/logprobs in each action.
        result={k:v for k,v in data.items() if k not in ('actions','external','summary_attempts')}
        result['actions']=[{k:a.get(k) for k in ('turn','raw_message_count','text','temperature','decoding_mode','ended_with_eos','policy_version')}|dict(prompt_tokens=len(a.get('prompt_ids',[])),generated_tokens=len(a.get('completion_ids',[]))) for a in data.get('actions',[])]
        return result


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                route=urlsplit(self.path);q=parse_qs(route.query)
                if route.path=='/':
                    content=(ROOT/'tools/rollout_viewer.html').read_bytes();mime='text/html; charset=utf-8'
                elif route.path=='/api/sources':content=json.dumps(store.sources()).encode();mime='application/json'
                elif route.path in ('/api/state','/api/episode'):
                    jid=q.get('job',[next(iter(store.jobs))])[0]
                    value=store.index(jid) if route.path=='/api/state' else store.episode(jid,q.get('key',[''])[0],q.get('raw',['0'])[0]=='1')
                    content=json.dumps(value,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(content)
            except (ValueError,FileNotFoundError) as error:self.send_error(404,str(error))
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:
                LOG.exception('Viewer request');self.send_error(500,'Viewer request failed; see server log')
        def log_message(self,*args):pass
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--job',action='append',type=Path,help='job.json; repeat to add previous runs')
    parser.add_argument('--sync-seconds',type=int,default=15);parser.add_argument('--no-sync',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    files=args.job or [ROOT/'hf_jobs/short_bank_rl/job.json',ROOT/'hf_jobs/short_retail_rl/job.json']
    jobs=[read_json(p) for p in files if p.exists()]
    if not jobs or any(not j or not re.fullmatch(r'[a-zA-Z0-9_-]+',j.get('job_id','')) for j in jobs):parser.error('Valid job metadata required')
    store=Store(jobs)
    if not args.no_sync:
        if args.sync_seconds<5:parser.error('Sync interval must be at least five seconds')
        threading.Thread(target=store.sync,args=(args.sync_seconds,),daemon=True).start()
    server=ThreadingHTTPServer((args.host,args.port),handler(store))
    print(f'Live rollout viewer: http://{args.host}:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
