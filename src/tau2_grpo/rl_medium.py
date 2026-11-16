"""Separate medium workflow bank and exact 60/20/20 retention curriculum."""
import json,hashlib
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from tau2_grpo.rl_config import path

@lru_cache(maxsize=2)
def load_medium(directory):
 root=path(directory);m=json.loads((root/'manifest.json').read_text())
 if not m.get('complete') or m.get('accepted')!=800:raise ValueError('Medium bank incomplete')
 for file,digest in m['files_sha256'].items():
  if hashlib.sha256((root/file).read_bytes()).hexdigest()!=digest:raise ValueError('Medium bank hash mismatch: '+file)
 rows={};pools={};worlds={}
 for split,n in [('train',640),('validation',80),('test',80)]:
  data=[json.loads(x) for x in (root/(split+'.jsonl')).read_text().splitlines()];assert len(data)==n
  pools[split]=[];worlds[split]=set()
  for r in data:
   if r['id'] in rows or r['split']!=split:raise ValueError('Duplicate or misplaced medium case')
   if not 4<=r['reference_actor_turns']<=7 or r['max_actor_turns']>8 or r['max_reference_prompt_tokens']+192>8192:raise ValueError('Medium budget invalid')
   rows[r['id']]=r;pools[split].append(r['id']);worlds[split].add(r['world_id'])
 if any(worlds[a]&worlds[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Medium customer split overlap')
 return root,rows,pools,json.loads((root/'catalog.json').read_text())

def medium_order(cfg,ids,rng):
 from tau2_grpo.rl_bank import load_bank,CLARIFICATION_FAMILIES
 rows=dict(load_bank(cfg.scenario_bank)[1]);medium=load_medium(cfg.medium_scenario_bank)[1];rows.update(medium);groups=defaultdict(lambda:defaultdict(list))
 for cid in ids:
  kind='medium' if cid in medium else 'clarification' if rows[cid]['family'] in CLARIFICATION_FAMILIES else 'retention';groups[kind][rows[cid]['family']].append(cid)
 if set(groups)!={'medium','clarification','retention'}:raise ValueError('Missing curriculum category')
 cycles={k:[] for k in groups};cases={};result=[];pattern=('medium','medium','clarification','medium','retention')
 for i in range(len(ids)):
  cat=pattern[i%5]
  if not cycles[cat]:cycles[cat]=sorted(groups[cat]);rng.shuffle(cycles[cat])
  family=cycles[cat].pop();key=(cat,family)
  if not cases.get(key):cases[key]=list(groups[cat][family]);rng.shuffle(cases[key])
  result.append(cases[key].pop())
 return result
