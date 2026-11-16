"""Execute matched, hint-free Tau2 continuations before/after HINT overfit."""
import argparse,json,gc,fcntl
from pathlib import Path
from dataclasses import replace
from tau2_grpo.opsd_retail import read_config,replay_to_recovery
from tau2_grpo.rl_rollout import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--overfit',required=True);p.add_argument('--actions',type=int,default=4);a=p.parse_args()
 run=Path(a.overfit);samples=json.loads((run/'samples.json').read_text());meta=json.loads((run/'config.json').read_text());source=Path(meta['source']);cfg,_=read_config(source/'launch-config.json')
 out=run/'live-continuations';out.mkdir(exist_ok=True)
 files={}
 for f in (source/'collection').glob('*.json'):
  x=json.loads(f.read_text());files[x['episode_id']]=f
 cases=[]
 for s in samples:
  cases.append({'candidate_id':s['candidate_id'],'task_id':s['task_id'],'seed':s['seed'],'source_file':str(files[s['episode_id']].resolve()),'recovery_turn':s['turn']})
 with open(Path.cwd()/'outputs/.retail-training.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  import torch
  from tau2_grpo.rl_policy import LocalPolicy
  results=[]
  for label,adapter in [('before',cfg.initial_adapter),('after',str((run/'checkpoint-12/policy').resolve()))]:
   local=replace(cfg,initial_adapter=adapter,temperature=0.)
   policy=LocalPolicy(local,trainable=False)
   for case in cases:
    result={'model':label,'case':case['candidate_id'],'actions':[],'reward':None,'scope':'up to four executed assistant actions from a verified saved state, without hints'}
    client=env=None
    try:
     client,controller,env,raw,_=replay_to_recovery(policy,local,case)
     result['replay_verified']=True;start=len(raw);done=False
     for step in range(a.actions):
      prepared=controller.prepare(raw,case['seed']+case['recovery_turn']+step)
      generated=policy.generate(prepared,192,42)
      raw.append({'role':'assistant','content':generated['text']})
      incoming,done,call=env.step(generated['text']);raw.extend(incoming)
      result['actions'].append({'text':generated['text'],'eos':generated['ended_with_eos'],'call':call,'observations':incoming})
      print(json.dumps({'model':label,'case':case['candidate_id'],'step':step,**result['actions'][-1]}),flush=True)
      if done:break
     result['done']=done;result['messages']=raw;result['continuation']=raw[start:]
     if done:
      try:result['reward'],result['simulation']=env.finish()
      except Exception as error:result['evaluation_error']=str(error)
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
     if env:
      try:env.close()
      except Exception:pass
     if client:client.close()
    results.append(result);atomic_json(out/'results.json',results)
    (out/'conversations.txt').write_text('\n\n'.join('MODEL: '+r['model']+' | CASE: '+r['case']+'\nREPLAY VERIFIED: '+str(r.get('replay_verified'))+'\n'+('\n\n'.join(m['role'].upper()+': '+str(m.get('content','')) for m in r.get('continuation',[])))+'\nERROR: '+str(r.get('error')) for r in results))
   del policy;gc.collect();torch.cuda.empty_cache()
 atomic_json(out/'summary.json',{'finished':True,'cases':len(cases),'models':['before','after'],'action_budget':a.actions,'results':len(results)})
if __name__=='__main__':main()
