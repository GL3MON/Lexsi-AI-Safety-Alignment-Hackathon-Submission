#!/usr/bin/env python3
"""Build reference-executed medium workflows; no model generation or paid jobs."""
import argparse,json,hashlib,shutil
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from tau2_grpo.rl_bank import load_bank
from tau2_grpo.medium_scenario_bank import FAMILIES,build_medium_case,validate_medium,reference_text,MediumEnvironment

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/retail-data/short-scenario-bank-3000');p.add_argument('--output',default='outputs/retail-data/medium-scenario-bank');p.add_argument('--world-limit',type=int,default=100);a=p.parse_args();root,rows,pools,catalog=load_bank(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained('outputs/rl-initialization/midturn-pilot-checkpoint6');tok.chat_template=(root/'chat_template.jinja').read_text();system=Path('templates/retail_apigen_sft_system.txt').read_text();counts=Counter();lengths=Counter();peak=0;worlds=set();digests={}
 for split in ['train','validation','test']:
  result=[]
  for cid in pools[split]:
   source=rows[cid];wid=source['world_id']
   if wid in worlds or int(wid.split('-')[1])>=a.world_limit:continue
   worlds.add(wid);world=json.loads((root/source['world_file']).read_text());dest=out/source['world_file'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/source['world_file'],dest)
   for family in FAMILIES:
    case=validate_medium(build_medium_case(world,catalog,system,family),world,catalog,tok);case.update(split=split,world_file=source['world_file']);result.append(case);counts[split]+=1;lengths[case['reference_actor_turns']]+=1;peak=max(peak,case['max_reference_prompt_tokens'])
    if wid=='world-0000':
     env=MediumEnvironment(case,world,catalog);history=env.reset()
     for step in case['routes'][0]:
      text=reference_text(step);incoming,_,_=env.step(text);history.append(dict(role='assistant',content=text));history.extend(incoming)
     (out/(family+'.txt')).write_text('\n\n'.join(m['role'].upper()+': '+m['content'] for m in history[1:])+'\n')
   print(json.dumps(dict(event='medium_world_verified',world=wid,cases=len(FAMILIES),accepted=sum(counts.values()))),flush=True)
  f=out/(split+'.jsonl');f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));digests[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 for name in ['catalog.json','chat_template.jinja']:shutil.copyfile(root/name,out/name)
 manifest=dict(version='medium-retail-v1',complete=True,accepted=sum(counts.values()),counts=dict(counts),families=list(FAMILIES),reference_actor_turns=dict(lengths),max_reference_prompt_tokens=peak,context_limit=8192,all_references_executed=True,files_sha256=digests,training_launched=False,note='Prepared medium workflows. RL loader integration and mixed-curriculum launch remain separate; original bank identity splits are preserved.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
