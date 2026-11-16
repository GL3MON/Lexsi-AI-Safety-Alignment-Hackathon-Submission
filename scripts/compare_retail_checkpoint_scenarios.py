"""Keep the paired baseline/SFT diagnostic report current until both finish."""
import json,time
from pathlib import Path
from tau2_grpo.rl_rollout import atomic_json
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'outputs/retail-eval/pre-retail-sft-glaive-rl17-task34'
sft=ROOT/'outputs/retail-eval/replay-sft-checkpoint50-task34'
out=ROOT/'outputs/retail-eval/glaive-rl17-vs-retail-sft50-task34';out.mkdir(exist_ok=True)
def update():
 pair=json.loads((base/'paired-first-turn.json').read_text());rows=[]
 for label,path in [('pre-SFT Glaive RL17',base),('retail SFT50',sft)]:
  p=path/'episode.json';e=json.loads(p.read_text()) if p.exists() else {}
  rows.append({'model':label,'reward':e.get('reward'),'complete':e.get('complete',False),'actor_turns':len(e.get('actions',[])),'stop_reason':e.get('stop_reason'),'error':e.get('error'),'transcript':str(path/'episode.conversation.txt')})
 report={'updated_at':time.time(),'paired_first_turn':pair,'episodes':rows,'all_finished':all(r['complete'] or r['error'] for r in rows),'interpretation':'The missing-identity hallucination exists before the new retail SFT. First-turn observations alone do not establish whether overall task performance regressed.'}
 atomic_json(out/'comparison.json',report)
 text=['TASK 34 — pre-SFT Glaive RL17 vs retail SFT50','Same exact initial prompt tokens, seed 100043, temperature 0.8, top-p 1.0.','', 'PRE-SFT FIRST RESPONSE:',pair['baseline_text'],'','POST-SFT FIRST RESPONSE:',pair['post_sft_text'],'','MODEL | REWARD | COMPLETE | ACTOR TURNS']
 text += [f"{r['model']} | {r['reward']} | {r['complete']} | {r['actor_turns']}" for r in rows]
 text += ['',report['interpretation'],'The Qwen customer simulator supplied an invalid order ID during the SFT rollout; distinguish that from agent-invented arguments.']
 (out/'comparison.txt').write_text('\n'.join(text)+'\n')
 from tau2_grpo.rl_reporting import write_episode_conversation
 for path in [base,sft]:
  p=path/'episode.json'
  if p.exists():write_episode_conversation(p,json.loads(p.read_text()))
 return report['all_finished']
if __name__=='__main__':
 while not update():time.sleep(10)
