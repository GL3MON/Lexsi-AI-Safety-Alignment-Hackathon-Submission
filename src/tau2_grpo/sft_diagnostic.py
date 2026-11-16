"""Single-decision free-generation checks on matched missing/supplied cases."""
import argparse,json,re
from pathlib import Path
from collections import defaultdict
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_context import ContextController
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.short_scenario_bank import matches,tools_from_world
from tau2_grpo.rl_rollout import atomic_json

def prepare_decision(tokenizer,prompt,cfg):
 # A controller tracks ONE conversation. Independent cases must never reuse it.
 def no_summary(*a,**kw):raise RuntimeError('Short diagnostic exceeded its context budget')
 controller=ContextController(tokenizer,no_summary,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
 prepared=controller.prepare(prompt,cfg.seed)
 from tau2_grpo.rl_context import student_messages
 if prepared['messages']!=student_messages(prompt):raise RuntimeError('Diagnostic rendered a different case history')
 return prepared

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',required=True);args=p.parse_args()
 from tau2_grpo.rl_policy import LocalPolicy
 from tau2.domains.retail.data_model import get_db
 cfg=RetailRLConfig(**json.loads(Path(args.config).read_text()));policy=LocalPolicy(cfg,trainable=False);root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);catalog={k:v.model_dump(mode='json') for k,v in get_db().products.items()}
 seen_prompts=set()
 for split in ['train','heldout']:
  rows=[json.loads(l) for l in (root/(split+'.jsonl')).read_text().splitlines()];results=[];directory=out/f'short-{args.phase}-{split}';directory.mkdir(exist_ok=True)
  for row in rows:
   prompt=row['messages'][:2];prepared=prepare_decision(policy.tokenizer,prompt,cfg)
   signature=tuple(prepared['prompt_ids'])
   if signature in seen_prompts:raise RuntimeError('Independent diagnostic cases rendered identical prompts')
   seen_prompts.add(signature)
   generated=policy.generate(prepared,cfg.response_tokens,cfg.seed);parse_error=None
   try:parsed=parse_calls(generated['text'])
   except ValueError as e:parsed=[];parse_error=str(e)
   protocol=generated.get('ended_with_eos',False) and parse_error is None and '<tool_response>' not in generated['text'] and not re.search(r'\b(?:user|customer|assistant)\s*:',generated['text'],re.I)
   correct=bool(protocol and matches(row['expected_action'],generated['text'],parsed,prompt[-1]['content']))
   predicted='call' if parsed else 'ask' if re.search(r'\?|\bplease\b|\bprovide\b',generated['text'],re.I) else 'other'
   messages=prompt+[dict(role='assistant',content=generated['text'])]
   if correct and row['category']=='call':
    world=json.loads((root/'worlds'/(row['world_id']+'.json')).read_text());tools=tools_from_world(world,catalog);c=parsed[0];observed=getattr(tools,c['name'])(**c['arguments']);messages.append(dict(role='tool',name=c['name'],content=json.dumps(dict(name=c['name'],content=observed))))
   action=dict(**prepared,**generated,raw_message_count=2,temperature=0.,turn=0,policy_version=0)
   record=dict(task_id=row['id'],pair_id=row['pair_id'],world_id=row['world_id'],field=row['field'],expected_category=row['category'],predicted_category=predicted,expected_action=row['expected_action'],reward=1. if correct else -1.,native_reward=None,short_task_success=correct,complete=True,diagnostic_decision_only=True,messages=messages,actions=[action],parse_error=parse_error,phase=args.phase)
   file=directory/(row['id']+'-bank.json');atomic_json(file,record);file.with_suffix('.txt').write_text('\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in messages))
   results.append(dict(task_id=row['id'],pair_id=row['pair_id'],field=row['field'],category=row['category'],success=correct,decision_type_correct=predicted==row['category'],complete=True,turns=1))
   print(json.dumps(dict(event='sft_decision_diagnostic',phase=args.phase,split=split,task_id=row['id'],success=correct,text=generated['text'])),flush=True)
   groups=defaultdict(list)
   for r in results:groups[r['pair_id']].append(r['success'])
   summary=dict(phase=args.phase,split=split,passed=sum(r['success'] for r in results),total=len(results),target=len(rows),complete=len(results)==len(rows),pairs_both_correct=sum(len(v)==2 and all(v) for v in groups.values()),pairs_complete=sum(len(v)==2 for v in groups.values()),categories={c:dict(passed=sum(r['success'] for r in results if r['category']==c),total=sum(r['category']==c for r in results)) for c in ['ask','call']},results=results,native_tau2_evaluation=False,diagnostic_decision_only=True)
   atomic_json(directory/'summary.json',summary)
  print('DIAGNOSTIC_SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
