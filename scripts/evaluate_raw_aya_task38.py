"""Untouched Aya weights, with identical evaluation template and task-38 settings."""
import hashlib,json,os
from pathlib import Path
from dataclasses import asdict,replace
import torch
from tau2_grpo.rl_config import ROOT,RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.rl import make_episode
from tau2_grpo.rl_reporting import write_episode_conversation
from tau2_grpo.rl_rollout import atomic_json
from scripts.evaluate_tau2_test_stages import InferenceOnlyPolicy
from transformers import AutoTokenizer

os.chdir(ROOT)
out=ROOT/'outputs/retail-eval/raw-aya-base-task38-native-eos';out.mkdir(parents=True,exist_ok=True)
if (out/'episode.json').exists():raise RuntimeError('Output exists; inspect before rerun')
raw=Path.home()/'.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324'
cfg=RetailRLConfig(model_id=str(raw),initial_adapter='',temperature=.7,top_p=.95,group_size=1,agent_prompt_file='templates/retail_eval_tool_preamble.txt')
class RawNativeStopPolicy(LocalPolicy):
 def generate(self,*args,**kwargs):
  original=self.tokenizer.eos_token
  self.tokenizer.eos_token=self.tokenizer.convert_ids_to_tokens(self.model.config.eos_token_id)
  try:return super().generate(*args,**kwargs)
  finally:self.tokenizer.eos_token=original
policy=RawNativeStopPolicy(cfg,adapter=False)
shared=AutoTokenizer.from_pretrained('outputs/hf-export/tiny-aya-global-function-calling-sft-525')
assert policy.tokenizer.get_vocab()==shared.get_vocab(),'Tokenizer vocabularies differ; cannot isolate weights'
policy.tokenizer=shared
atomic_json(out/'config.json',{'settings':asdict(cfg),'task_id':'38','raw_weight_revision':raw.name,'raw_weights_no_adapters':True,'eos_token_id':policy.model.config.eos_token_id,'shared_tokenizer':'outputs/hf-export/tiny-aya-global-function-calling-sft-525','shared_template_sha256':hashlib.sha256(shared.chat_template.encode()).hexdigest(),'note':'Raw Aya weights with the same evaluation chat template as trained checkpoints; isolates weight changes. One sampled rollout, not a statistically conclusive regression test.'})
from loguru import logger
logger.remove();logger.add(__import__('sys').stderr,level='ERROR')
import wandb
wb=wandb.init(project='tau2-retail',mode='online',name='raw-aya-base-task38-native-eos',dir=str(out),job_type='heldout-test-evaluation',config=json.loads((out/'config.json').read_text()))
atomic_json(out/'runtime.json',{'wandb_url':wb.url})
try:
 episode=make_episode(InferenceOnlyPolicy(policy),cfg,'38',42,out/'episode.json')
 transcript=write_episode_conversation(out/'episode.json',episode)
 summary={'task_id':'38','model':'CohereLabs/tiny-aya-global (raw weights)','reward':episode['reward'],'complete':episode['complete'],'actor_turns':len(episode['actions']),'tool_calls':sum(len(m.get('tool_calls') or []) for m in episode['messages'] if m['role']=='assistant'),'stop_reason':episode.get('stop_reason'),'error':episode.get('error'),'checks':episode['check_report']['checks'],'transcript':str(transcript)}
 atomic_json(out/'summary.json',summary);print('RESULT',json.dumps(summary),flush=True)
 wb.log({'reward':episode['reward'],'actor_turns':len(episode['actions'])})
finally:wb.finish()
