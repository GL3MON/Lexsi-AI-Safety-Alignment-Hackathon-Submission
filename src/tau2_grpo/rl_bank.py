"""Verified scenario-bank loading; hidden routes never enter actor messages."""
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from tau2_grpo.rl_config import path

@lru_cache(maxsize=2)
def load_bank(directory):
    root=path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('accepted')!=3000:
        raise ValueError('Scenario bank is incomplete')
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Scenario bank hash mismatch: '+name)
    rows={};pools={};worlds={}
    for split,expected in [('train',2400),('validation',300),('test',300)]:
        data=[json.loads(line) for line in (root/(split+'.jsonl')).read_text().splitlines()]
        if len(data)!=expected:raise ValueError('Incorrect split count')
        pools[split]=[r['id'] for r in data];worlds[split]={r['world_id'] for r in data}
        for r in data:
            if r['id'] in rows or r['split']!=split:raise ValueError('Duplicate/misplaced case')
            if r['max_actor_turns']>4 or r['max_reference_prompt_tokens']+192>8192:raise ValueError('Case exceeds short budget')
            rows[r['id']]=r
    if any(worlds[a]&worlds[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Customer worlds overlap')
    return root,rows,pools,json.loads((root/'catalog.json').read_text())

def bank_task_ids(cfg,partition='rl'):
    _,rows,pools,_=load_bank(cfg.scenario_bank)
    ids=list(pools['train' if partition=='rl' else partition])
    if cfg.bank_view=='midturn_mix':
        from tau2_grpo.rl_midturn import MID_PREFIX
        ids += [MID_PREFIX+cid for cid in ids if rows[cid]['family'] in CLARIFICATION_FAMILIES]
    if cfg.bank_view=='medium_mix':
        from tau2_grpo.rl_medium import load_medium
        ids += list(load_medium(cfg.medium_scenario_bank)[2]['train' if partition=='rl' else partition])
    return ids

def bank_environment(cfg,case_id):
    from tau2_grpo.short_scenario_bank import BankEnvironment
    root,rows,_,catalog=load_bank(cfg.scenario_bank)
    from tau2_grpo.rl_midturn import MID_PREFIX,derive_case
    source_id=case_id.removeprefix(MID_PREFIX)
    if case_id.startswith(MID_PREFIX) and cfg.bank_view!='midturn_mix':raise ValueError('Midturn state outside selected curriculum')
    cls=BankEnvironment
    if cfg.bank_view=='medium_mix':
        from tau2_grpo.rl_medium import load_medium
        from tau2_grpo.medium_scenario_bank import MediumEnvironment
        mr,mrows,_,mcatalog=load_medium(cfg.medium_scenario_bank)
        if source_id in mrows:root,rows,catalog,cls=mr,mrows,mcatalog,MediumEnvironment
    case=rows[source_id]
    world=json.loads((root/case['world_file']).read_text())
    if case_id.startswith(MID_PREFIX):case=derive_case(case,world,catalog)
    if cfg.max_actions<case['max_actor_turns']:raise ValueError('Insufficient actor turn budget')
    if case['messages'][0]['content']!=path(cfg.system_prompt_file).read_text():raise ValueError('Bank system prompt differs')
    if (root/'chat_template.jinja').read_text()!=path(cfg.chat_template_file).read_text():raise ValueError('Bank template differs')
    world=json.loads((root/case['world_file']).read_text())
    return cls(case,world,catalog)

CLARIFICATION_FAMILIES=('missing_identity','missing_first_name','missing_last_name','missing_zip','partial_first_name','partial_last_name','partial_zip','failed_email')

def clarification_order(cfg,ids,rng):
    _,rows,_,_=load_bank(cfg.scenario_bank);families=defaultdict(list)
    for cid in ids:families[rows[cid]['family']].append(cid)
    categories={'clarification':[k for k in sorted(families) if k in CLARIFICATION_FAMILIES], 'retention':[k for k in sorted(families) if k not in CLARIFICATION_FAMILIES]}
    if not all(categories.values()):raise ValueError('Clarification curriculum needs both categories')
    family_cycles={k:[] for k in categories};case_cycles={k:[] for k in families};result=[]
    for i in range(len(ids)):
        category='retention' if i%3==2 else 'clarification'
        if not family_cycles[category]:
            family_cycles[category]=list(categories[category]);rng.shuffle(family_cycles[category])
        family=family_cycles[category].pop()
        if not case_cycles[family]:
            case_cycles[family]=list(families[family]);rng.shuffle(case_cycles[family])
        result.append(case_cycles[family].pop())
    return result

def balanced_order(cfg,ids,rng):
    if cfg.bank_sampling=='medium_mix':
        from tau2_grpo.rl_medium import medium_order
        return medium_order(cfg,ids,rng)
    if cfg.bank_sampling=='midturn_mix':
        from tau2_grpo.rl_midturn import mixed_order
        return mixed_order(cfg,ids,rng)
    if cfg.bank_sampling=='clarification_mix':return clarification_order(cfg,ids,rng)
    _,rows,_,_=load_bank(cfg.scenario_bank);families=defaultdict(list)
    for case_id in ids:families[rows[case_id]['family']].append(case_id)
    keys=sorted(families)
    for group in families.values():rng.shuffle(group)
    result=[]
    while any(families.values()):
        rng.shuffle(keys)
        result.extend(families[key].pop() for key in keys if families[key])
    return result

def validation_cases(cfg):
    _,rows,pools,_=load_bank(cfg.scenario_bank)
    requested={f'world-{int(w):04d}' for w in cfg.bank_validation_worlds.split(',') if w.strip()}
    if requested:
        available={rows[cid]['world_id'] for cid in pools['validation']}
        if not requested <= available:raise ValueError('Requested worlds outside validation partition')
        ids=sorted(cid for cid in pools['validation'] if rows[cid]['world_id'] in requested)
    else:
        selected={}
        for case_id in sorted(pools['validation']):selected.setdefault(rows[case_id]['family'],case_id)
        ids=[selected[k] for k in sorted(selected)]
    if cfg.bank_view=='midturn_mix':
        from tau2_grpo.rl_midturn import MID_PREFIX
        ids += [MID_PREFIX+cid for cid in ids.copy() if rows[cid]['family'] in CLARIFICATION_FAMILIES]
    if cfg.bank_view=='medium_mix':
        from tau2_grpo.rl_medium import load_medium
        _,mrows,mpools,_=load_medium(cfg.medium_scenario_bank)
        chosen=requested or {rows[ids[0]]['world_id']}
        ids += sorted(cid for cid in mpools['validation'] if mrows[cid]['world_id'] in chosen)
    return ids


def case_family(cfg,case_id):
    from tau2_grpo.rl_midturn import MID_PREFIX
    if cfg.bank_view=='medium_mix':
        from tau2_grpo.rl_medium import load_medium
        rows=load_medium(cfg.medium_scenario_bank)[1]
        if case_id in rows:return 'medium_'+rows[case_id]['family']
    return ('after_reply_' if case_id.startswith(MID_PREFIX) else '')+load_bank(cfg.scenario_bank)[1][case_id.removeprefix(MID_PREFIX)]['family']
