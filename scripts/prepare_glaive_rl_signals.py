"""Preserve reserved RL prompts; add controlled, auditable decision fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
from tau2_grpo.glaive_signals import score_decision
from tau2_grpo.glaive_prompts import system_prompt,check_system


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/tiny-aya-glaive/rl_train.jsonl');p.add_argument('--output',default='outputs/glaive-rl-signals-v2');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    rows=[]; counts={}
    def add(row):
        rows.append(row);kind=row['scenario'];counts[kind]=counts.get(kind,0)+1
    for line in Path(a.source).open():
        base=json.loads(line);base['source_id']=base['id'];base['scenario']='original'
        base['prompt'][0]['content']=system_prompt(base['tools'])
        evidence={};customer='\n'.join(m['content'] for m in base['prompt'] if m['role']=='user')
        for c in base['reward_reference']['tool_calls']:
            evidence[c['name']]={k:v for k,v in c['arguments'].items() if isinstance(v,str) and v and v in customer}
        base['reward_reference']['literal_arguments']=evidence;add(base)
        call=json.loads(json.dumps(base['reward_reference']['tool_calls'][0]))
        token=hashlib.sha256(base['id'].encode()).hexdigest()
        first=('Nerina','Tavish','Ilyana','Zevan','Amara','Leorin','Keira','Darian')[int(token[:2],16)%8]
        last=('Varela','Okafor','Nakamura','Bellamy','Qureshi','Solberg','Menon','Valente')[int(token[2:4],16)%8]
        n=int(token[4:8],16)
        emails=(f'{first.lower()}.{last.lower()}+orders@post.example.org',
                f'{first[0]}_{last}.receipts@accounts.example.net',
                f'{last.lower()}-{first.lower()}@mail.example.com',
                f'{first.lower()}{n%97:02d}.{last.lower()}@support.example.org')
        for key,value in list(call['arguments'].items()):
            if not isinstance(value,str):continue
            lower=key.lower()
            if 'email' in lower:call['arguments'][key]=emails[int(token[8:10],16)%4]
            elif lower in ('first_name','firstname'):call['arguments'][key]=first
            elif lower in ('last_name','lastname','surname'):call['arguments'][key]=last
            elif lower in ('full_name','fullname','customer_name','user_name'):call['arguments'][key]=first+' '+last
        schema=next(t['function']['parameters'] for t in base['tools'] if t['function']['name']==call['name'])
        eligible=[k for k in schema.get('required',[]) if isinstance(call['arguments'].get(k),str) and call['arguments'][k] and len(call['arguments'][k])>=3]
        if not eligible:continue
        field=eligible[0];gold=call['arguments'][field]
        # Deliberately controlled fixtures: no reliance on removing every semantic
        # clue from a free-form original request. All listed values are authoritative.
        fields={k:v for k,v in call['arguments'].items() if k!=field}
        opening=f"I need help with {call['name'].replace('_',' ')}. These are the details I have: {json.dumps(fields,ensure_ascii=False)}. I have not provided {field.replace('_',' ')} yet."
        prompt=[base['prompt'][0],{'role':'user','content':opening}]
        def variant(kind,messages,ref):
            return {**base,'id':base['id']+':'+kind,'scenario':kind,'prompt':messages,'reward_reference':ref,'synthetic':True}
        clarify={'decision':'clarify','missing_field':field,'missing_field_aliases':[field,field.replace('_',' ')],'assistant':f"Could you provide the {field.replace('_',' ')}?"}
        add(variant('missing_required',prompt,clarify))
        # Known reference after clarification: exact values are retained by code.
        follow=prompt+[{'role':'assistant','content':clarify['assistant']},{'role':'user','content':f"The {field.replace('_',' ')} is {json.dumps(gold,ensure_ascii=False)}."}]
        ref={'tool_calls':[call],'assistant':'<tool_call>'+json.dumps(call)+'</tool_call>','literal_arguments':{call['name']:call['arguments']}}
        add(variant('clarification_followup',follow,ref))
        wrong='OUTDATED_VALUE_'+hashlib.sha256(base['id'].encode()).hexdigest()[:8]
        old={**call,'arguments':{**call['arguments'],field:wrong}}
        correction=[base['prompt'][0],{'role':'user','content':f"Please help with {call['name'].replace('_',' ')} using these details: {json.dumps(old['arguments'])}."},{'role':'assistant','content':f"Please confirm the {field.replace('_',' ')}."},{'role':'user','content':f"Correction: use {json.dumps(gold,ensure_ascii=False)} for {field.replace('_',' ')} instead of {json.dumps(wrong)}. All other details remain the same."}]
        add(variant('corrected_value',correction,ref))
        # A synthetic lookup fixture supplies the exact argument through a tool
        # result. Its schema is included; no claim of an executable Glaive API.
        lookup='resolve_reference_'+field
        if any(t['function']['name']==lookup for t in base['tools']):continue
        tools=base['tools']+[{'type':'function','function':{'name':lookup,'description':'Read the verified reference value for this request.','parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False}}}]
        system={'role':'system','content':system_prompt(tools)}
        history=[system,{'role':'user','content':f"Please help with {call['name'].replace('_',' ')}. Known details: {json.dumps(fields)}. Use the verified lookup result for {field.replace('_',' ')}."},{'role':'assistant','content':'<tool_call>'+json.dumps({'name':lookup,'arguments':{}})+'</tool_call>'},{'role':'user','content':'<tool_response>'+json.dumps({'name':lookup,'content':{field:gold}})+'</tool_response>'}]
        row=variant('tool_result_copy',history,ref);row['tools']=tools;add(row)
    failures=[]
    for r in rows:
        check_system(r['prompt'][0]['content'],r['tools'])
        scored=score_decision(r['reward_reference']['assistant'],r)
        if scored['reward']<.94:failures.append({'id':r['id'],'score':scored})
    if failures:raise ValueError(json.dumps(failures[:3]))
    (root/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'source':a.source,'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),'counts':counts,'total':len(rows),'self_check_failures':len(failures),'split':'reserved RL only; synthetic variants retain parent group_id','limitations':['Clarification scoring is a targeted-question heuristic, not proof of helpfulness.','Synthetic tool results are fixtures, not live Glaive APIs.','Controlled wording is intentionally explicit; evaluate separately on natural held-out prompts.','Qwen does not supply reward labels.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
