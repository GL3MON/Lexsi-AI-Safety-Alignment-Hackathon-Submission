"""Audit deterministic reward coverage; Qwen proposes evidence, never grades."""
import json
from pathlib import Path
from collections import Counter
from openai import OpenAI


def main():
    rows=[json.loads(s) for s in Path('outputs/tiny-aya-glaive/rl_train.jsonl').open()]
    stats=Counter();candidates=[]
    for row in rows:
        user='\n'.join(m['content'] for m in row['prompt'] if m['role']=='user')
        for call in row['reward_reference']['tool_calls']:
            for key,value in call['arguments'].items():
                stats['argument_values']+=1
                if isinstance(value,str):
                    stats['string_values']+=1
                    if value in user:
                        stats['exact_string_in_user']+=1
                        if value and user.count(value)==1:
                            stats['unique_literal_string']+=1
                            if len(value)>=5 and len(candidates)<5:
                                candidates.append((row,key,value,user))
    client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=90,max_retries=0)
    probes=[]
    for row,key,value,user in candidates:
        response=client.chat.completions.create(model='qwen3.5-4b',messages=[{'role':'system','content':'Extract literal evidence only. Return JSON {"evidence": "exact substring from customer message"}. No explanation.'},{'role':'user','content':json.dumps({'customer':user,'argument':key,'reference_value':value})}],temperature=0,max_tokens=128,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        text=response.choices[0].message.content
        try:
            evidence=json.loads(text)['evidence']
            verified=isinstance(evidence,str) and evidence==value and evidence in user
        except (ValueError,KeyError,TypeError): evidence=None;verified=False
        # Deterministic counterfactual: delete the sole literal reference value.
        # Semantic sufficiency of other text remains a separate review requirement.
        probes.append({'id':row['id'],'argument':key,'value':value,'customer':user,'qwen_output':text,'literal_evidence_verified':verified,'masked_customer':user.replace(value,'[NOT PROVIDED]'),'note':'Literal deletion is mechanically verified, but alternative evidence/derivability must be checked before labeling this a missing-slot example.'})
    report={'coverage':dict(stats),'qwen_evidence_probes':probes,'qwen_passes':sum(p['literal_evidence_verified'] for p in probes),'probe_count':len(probes)}
    out=Path('outputs/diagnostics/glaive-rl-preparation');out.mkdir(parents=True,exist_ok=True)
    (out/'verifiable-signals.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
