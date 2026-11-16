"""Audit the existing held-out Glaive RL decisions and reference reward."""
import json
from collections import Counter
from pathlib import Path
from tau2_grpo.glaive_rewards import score_calls
from tau2_grpo.prepare_function_calling import tool_validators


def main():
    root=Path('outputs/tiny-aya-glaive')
    rows=[json.loads(line) for line in (root/'rl_train.jsonl').open()]
    errors=[]; counts=Counter()
    for row in rows:
        try:
            calls=row['reward_reference']['tool_calls']
            validators=tool_validators(row['tools'])
            for c in calls: validators[c['name']].validate(c['arguments'])
            result=score_calls(row['reward_reference']['assistant'],calls)
            assert result['reward']==1 and result['exact_call_set']
            assert all(m['role']!='assistant' for m in row['prompt'])
            counts[len(calls)]+=1
        except Exception as exc: errors.append({'id':row['id'],'error':str(exc)})
    groups={r['group_id'] for r in rows}
    overlaps={}
    for name in ('sft_train','sft_val','test'):
        others={json.loads(line)['group_id'] for line in (root/(name+'.jsonl')).open()}
        overlaps[name]=len(groups & others)
    report={'rl_source':str(root/'rl_train.jsonl'),'prompts':len(rows),'reference_call_counts':dict(counts),'invalid_references':errors,'group_overlap':overlaps,'reward':'one-to-one exact reference-call matches / generated calls; zero for no calls or malformed output','notes':['Reference calls stay outside the prompt.','Reference equality is a proxy for task correctness; valid alternative calls may score zero.','This split contains call-required initial decisions; it cannot teach missing-information clarification by itself.','Recall and F1 are logged separately; precision alone does not require all expected calls.']}
    out=Path('outputs/diagnostics/glaive-rl-preparation');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors or any(overlaps.values()): raise SystemExit(1)

if __name__=='__main__':main()
