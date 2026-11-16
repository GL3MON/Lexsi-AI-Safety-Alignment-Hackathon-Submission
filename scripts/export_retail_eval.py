"""Export readable Tau2 evaluation conversations and a compact results index."""
import argparse,json,time
from pathlib import Path
from tau2_grpo.rl_reporting import write_episode_conversation,episode_check_report

def export(directory):
 directory=Path(directory);rows=[]
 for file in sorted(directory.glob('task-*.json')):
  try:episode=json.loads(file.read_text())
  except (OSError,ValueError):continue
  if not episode.get('complete'):continue
  target=write_episode_conversation(file,episode);report=episode_check_report(episode)
  db=next((c['status'] for c in report['checks'] if c['category']=='DB'),'not_evaluated')
  rows.append((episode,report,db,target))
 lines=['TAU2 VALIDATION — '+directory.name,'',f'Completed episodes: {len(rows)} / 10',f'Reward 1: {sum(e.get("reward")==1 for e,_,_,_ in rows)}','',
  'TASK | REWARD | DB CHECK | ACTOR TURNS | CHECK COUNTS','-'*80]
 for e,report,db,target in rows:
  lines.append(f"{e['task_id']} | {e.get('reward')} | {db} | {report['turn_counts']['actor_turns']} | {json.dumps(report['check_counts'])}")
 lines+=['','Readable conversations include exact assistant outputs, tool results, individual checks, and context summaries.','']
 for e,report,db,target in rows:
  lines+=[f"TASK {e['task_id']}: {target.name}"]
  for check in report['checks']:
   if check['status']!='pass':lines.append('  '+check['status'].upper()+': '+check['description'])
 (directory/'EVALUATION.txt').write_text('\n'.join(lines)+'\n')
 return len(rows)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  count=export(a.directory)
  if not a.watch or (Path(a.directory)/'summary.json').exists():break
  time.sleep(10)
 print('Exported',count,'episodes')
