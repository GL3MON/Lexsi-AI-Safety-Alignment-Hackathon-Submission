"""Replace verbose diagnostic objects with validated short Qwen instructions."""
import json,re
from pathlib import Path
from openai import OpenAI
from tau2_grpo.hint_distillation import teacher_messages
from tau2_grpo.glaive_episode import episode_rows


def main():
    source=Path('outputs/diagnostics/hint-teacher-token-probe-v2/prepared_turns.json')
    turns=json.loads(source.read_text());out=Path('outputs/diagnostics/hint-short-instructions');out.mkdir(parents=True,exist_ok=True)
    baselines=json.load(open('outputs/diagnostics/opsd-per-turn-replay-validated/baseline.json'))
    client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=120,max_retries=1)
    system=Path('templates/opsd_hinter_system.txt').read_text();prepared=[];attempts=[]
    rows={r['id']:r for r in episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()])}
    for t in turns:
        e=next(x['episode'] for x in baselines if x['row_id']==t['row_id'] and x['seed']==42)
        row=rows[t['row_id']];ref=row['reward_reference'] if t['turn']==1 else row.get('final_reference',row['reward_reference'])
        ask=ref.get('decision')=='clarify'
        contract={'decision':'ask_missing_field' if ask else 'copy_available_value','field':ref.get('missing_field') if ask else list(ref['tool_calls'][0]['arguments']),'source':'customer must supply missing field' if ask else 'authoritative earlier customer message or successful tool result'}
        values=[v for c in ref.get('tool_calls',[]) for v in c['arguments'].values() if isinstance(v,str) and len(v)>=4]
        if not ask:
            contract['source']='previous successful tool result' if t['history_before_turn'][-1]['content'].startswith('<tool_response>') else 'latest customer reply'
            contract['field']='email' if 'email' in ref['tool_calls'][0]['arguments'] else 'text'
        payload={'authoritative_turn_contract':contract,'instruction':'Write only the short action instruction from this verified contract. Do not add a diagnosis or quote any values or specific characters.'}
        request=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        for retry in range(3):
            r=client.chat.completions.create(model='qwen3.5-4b',messages=request,temperature=0,max_tokens=180,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            raw=r.choices[0].message.content
            try:hint=json.loads(raw)['hint']
            except (ValueError,KeyError,TypeError):hint=''
            valid=isinstance(hint,str) and 0<len(hint.split())<=24 and re.match(r'^(Ask|Copy|Use|Preserve|Retrieve|Keep|Call|Read|Request|Select|Wait|Return|Provide|Respond)\b',hint) and not re.search(r'\bthe agent\b|violated|failed to|<tool_call>',hint,re.I)
            valid=valid and all(ord(char)<128 for char in hint) and (hint.startswith('Ask ') if ask else hint.startswith('Copy ')) and not re.search(r'@|\bdouble\b|\bconfirm\b',hint,re.I) and not any(v.lower() in hint.lower() for v in values)
            attempts.append({'row_id':t['row_id'],'turn':t['turn'],'response':raw,'valid':bool(valid)})
            if valid:break
            request.extend([{'role':'assistant','content':raw},{'role':'user','content':'Rewrite as one imperative instruction of at most 24 words following authoritative_turn_contract. Start Ask for missing fields, Copy for available values. No quoted values, confirmations, or invented character details. Return only {"hint":"..."}.'}])
        if not valid:
            (out/'hinter_attempts.json').write_text(json.dumps(attempts,indent=2,ensure_ascii=False))
            raise ValueError('No valid short hint: '+repr(hint))
        item={**t,'hint':hint,'teacher_history':teacher_messages(t['history_before_turn'],t['observation'],t['episode_reward'],hint)}
        prepared.append(item);print(t['row_id'],t['turn'],hint,flush=True)
        (out/'prepared_turns.json').write_text(json.dumps(prepared,indent=2,ensure_ascii=False));(out/'hinter_attempts.json').write_text(json.dumps(attempts,indent=2,ensure_ascii=False))
        (out/'hints.txt').write_text('\n\n'.join(x['row_id']+' | TURN '+str(x['turn'])+'\n'+x['hint'] for x in prepared))

if __name__=='__main__':main()
