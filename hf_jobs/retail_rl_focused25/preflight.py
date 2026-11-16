import json,hashlib
from pathlib import Path
from loguru import logger
logger.remove();logger.add(__import__('sys').stderr,level='ERROR')
from openai import OpenAI
from transformers import AutoTokenizer
from tau2_grpo.rl_config import load_settings,task_ids
from tau2_grpo.rl_tau2 import RetailEnvironment,ExternalChat
cfg=load_settings('hf_jobs/retail_rl_focused25/smoke-config.json');ids=task_ids(cfg)
tok=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=180,max_retries=0);external=ExternalChat(client,tok,cfg,[]);env=RetailEnvironment(ids[0],42,cfg,external)
try:
 messages=env.reset();system=Path(cfg.system_prompt_file).read_text();assert messages[0]['content']==system
 assert 'get_item_details' not in env.schemas and 'cancel_pending_order' in env.schemas
 report=dict(ready=True,task_id=ids[0],rl_partition_tasks=len(ids),exact_system_prompt=True,system_sha256=hashlib.sha256(system.encode()).hexdigest(),tool_names=sorted(env.schemas),max_actions=cfg.max_actions,rollouts=cfg.group_size,checkpoint=25)
 here=Path('hf_jobs/retail_rl_focused25/artifacts');here.mkdir(exist_ok=True);(here/'environment-preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:env.close();client.close()
