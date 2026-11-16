"""Descriptive turn-level tool error and context-length analysis."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.opsd_retail import tool_error
ROOT=Path('outputs');OUT=ROOT/'diagnostics/retail-horizon'

def describe(rows):
 n=len(rows);calls=sum(r['tool_attempt'] for r in rows)
 return {'turns':n,'tool_attempts':calls,'malformed':sum(r['malformed'] for r in rows),
  'malformed_percent_of_tool_attempts':round(100*sum(r['malformed'] for r in rows)/calls,1) if calls else None,
  'tool_errors':sum(r['tool_error'] for r in rows),'repeat_failed':sum(r['repeat_failed'] for r in rows),
  'repeat_call':sum(r['repeat_call'] for r in rows),'truncated':sum(r['truncated'] for r in rows)}

def analyze(files):
 rows=[];episodes=[];skate=[]
 for f in files:
  try:e=json.loads(f.read_text())
  except (OSError,ValueError):continue
  if not e.get('complete'):continue
  actions=e.get('actions') or [];messages=e.get('messages') or []
  previous_signature=None;previous_error=False;previous_summaries=0;erows=[]
  for index,a in enumerate(actions):
   start=a['raw_message_count']+1;end=actions[index+1]['raw_message_count'] if index+1<len(actions) else len(messages)
   observations=messages[start:end];text=a.get('text','');malformed=False
   try:calls=parse_calls(text)
   except ValueError:calls=[];malformed=True
   runtime=' '.join(str(m.get('content','')) for m in observations if m.get('name')=='__runtime__')
   malformed=malformed or 'invalid_tool_call_json' in runtime
   signature=json.dumps(calls,sort_keys=True) if calls else text.strip()
   attempted=bool(calls or malformed or '<tool_call>' in text)
   error=any(tool_error(m) for m in observations)
   summary=a.get('summary_count',0)
   row={'file':str(f),'task':str(e['task_id']),'turn':index+1,'prompt_tokens':len(a.get('prompt_ids',[])),
    'completion_tokens':len(a.get('completion_ids',[])),'summary_count':summary,'after_summary':summary>previous_summaries,
    'tool_attempt':attempted,'malformed':malformed,'tool_error':error,
    'repeat_call':bool(attempted and signature==previous_signature),
    'repeat_failed':bool(attempted and signature==previous_signature and previous_error),
    'truncated':not a.get('ended_with_eos',True),'text':text[:350]}
   rows.append(row);erows.append(row);previous_signature=signature;previous_error=error;previous_summaries=summary
  episodes.append({'file':str(f),'task':e['task_id'],'reward':e.get('reward'),'turns':len(actions),
    'first_malformed_turn':next((r['turn'] for r in erows if r['malformed']),None),
    'first_tool_error_turn':next((r['turn'] for r in erows if r['tool_error']),None)})
  if any('skateboard' in r['text'].lower() and r['malformed'] for r in erows):skate.extend(erows)
 buckets={label:describe([r for r in rows if low<=r['turn']<=high]) for label,low,high in [('1-5',1,5),('6-10',6,10),('11-20',11,20),('21+',21,999)]}
 prompts={label:describe([r for r in rows if low<=r['prompt_tokens']<high]) for label,low,high in [('<6000',0,6000),('6000-6999',6000,7000),('7000+',7000,99999)]}
 long_files={e['file'] for e in episodes if e['turns']>10}
 summary={'episodes':len(episodes),'reward_1':sum(e['reward']==1 for e in episodes),'overall':describe(rows),'by_turn':buckets,'by_prompt_tokens':prompts,
  'before_any_summary':describe([r for r in rows if r['summary_count']==0]),'with_summary':describe([r for r in rows if r['summary_count']>0]),
  'immediately_after_summary':describe([r for r in rows if r['after_summary']]),
  'same_long_episodes_first_10':describe([r for r in rows if r['file'] in long_files and r['turn']<=10]),
  'same_long_episodes_after_10':describe([r for r in rows if r['file'] in long_files and r['turn']>10]),'episode_details':episodes,'skateboard_timeline':skate}
 return summary

sources={'collection':list((ROOT/'retail-hint/run-20261111T205531-62tasks/collection').glob('episode-*.json')),
 'eval_step50':list((ROOT/'retail-hint-training/run-20261112T082253/evaluation/step-000050').glob('task-*.json'))}
result={name:analyze(files) for name,files in sources.items()}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(result,indent=2))
for name,x in result.items():
 print(name,'episodes',x['episodes'],'successes',x['reward_1'])
 for k in ('by_turn','by_prompt_tokens','before_any_summary','with_summary','same_long_episodes_first_10','same_long_episodes_after_10'):
  print(k,json.dumps(x[k]))
 print('skateboard error timeline',json.dumps([{k:r[k] for k in ('task','turn','prompt_tokens','summary_count','after_summary','malformed','repeat_failed','text')} for r in x['skateboard_timeline'] if r['malformed']]))
