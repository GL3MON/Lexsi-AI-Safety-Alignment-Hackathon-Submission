"""Does paper-style privileged context produce a verifiably better action?"""
import json
import argparse
from pathlib import Path
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows
from tau2_grpo.glaive_signals import score_decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared-dir',default='outputs/diagnostics/hint-teacher-token-probe-v2');parser.add_argument('--output-dir',default='outputs/diagnostics/hint-teacher-correction-test');args=parser.parse_args()
    torch.set_num_threads(8);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    turns=json.loads((Path(args.prepared_dir)/'prepared_turns.json').read_text())
    rows={r['id']:r for r in episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals/train.jsonl').open()])}
    cases=[t for t in turns if not(t['row_id'].endswith('missing_required') and t['turn']==1 and 'email' in rows[t['row_id']]['reward_reference'].get('missing_field',''))]
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-e864eb346e/checkpoint-16/policy',temperature=.7,top_p=1.,response_tokens=256,max_actions=10)
    policy=LocalPolicy(cfg);results=[]
    def save():
        (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        summary=[]
        for t in cases:
            summary.append({'row_id':t['row_id'],'turn':t['turn'],**{arm:{'correct':sum(r['correct'] for r in results if r['row_id']==t['row_id'] and r['arm']==arm),'total':sum(r['row_id']==t['row_id'] and r['arm']==arm for r in results)} for arm in ('unhinted','paper_teacher')}})
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'conversations.txt').write_text('\n\n'.join('CASE: '+r['row_id']+' | TURN: '+str(r['turn'])+' | ARM: '+r['arm']+' | SEED: '+str(r['seed'])+' | CORRECT: '+str(r['correct'])+'\n\n'+'\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in r['messages'][1:]) for r in results))
    for t in cases:
        row=rows[t['row_id']]
        if row['scenario']=='missing_required' and t['turn']>1:row={**row,'reward_reference':row['final_reference']}
        for arm in ('unhinted','paper_teacher'):
            messages=t['history_before_turn'] if arm=='unhinted' else t['teacher_history']
            ids=policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            for seed in range(42,46):
                generated=policy.generate({'prompt_ids':ids},256,seed);details=score_decision(generated['text'],row)
                correct=bool(generated['ended_with_eos'] and (details.get('clarification_correct') if row['reward_reference'].get('decision')=='clarify' else details.get('exact_call_set') and details.get('schema_validity')==1 and details.get('format_valid')))
                results.append({'row_id':t['row_id'],'turn':t['turn'],'arm':arm,'seed':seed,'correct':correct,'score':details,'output':generated['text'],'messages':messages+[{'role':'assistant','content':generated['text']}],'hint':t['hint'] if arm=='paper_teacher' else None});save();print(arm,t['turn'],seed,correct,repr(generated['text']),flush=True)
    (out/'metadata.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'seeds':[42,43,44,45],'temperature':.7,'top_p':1.,'training_updates':0,'metric':'Correct next action or targeted clarification; not complete episode success.','method':'Same frozen Aya; Qwen generated completed-rollout diagnoses; teacher gets pre-turn history plus observation, outcome and diagnosis.','limitation':'Generating a new response is an auxiliary diagnostic, not the paper token-scoring training objective.'},indent=2))

if __name__=='__main__':main()
