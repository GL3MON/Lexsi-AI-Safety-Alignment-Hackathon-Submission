"""Short real-tool GRPO curriculum; no hosted customer or Hub output commits."""
import os,sys,json,subprocess,tarfile,hashlib,base64,zlib,traceback
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download,snapshot_download
work=Path('/workspace/short-retail-rl');work.mkdir(parents=True,exist_ok=True)
store=Path(os.environ['RL_CHECKPOINT_DIR']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
try:
 run([sys.executable,'-m','pip','install','-q','uv'])
 repo='GL3MON/tau2-retail-focused25-rl-input';payload=hf_hub_download(repo,'payload.tar.gz',repo_type='dataset');meta=json.loads(Path(hf_hub_download(repo,'payload-manifest.json',repo_type='dataset')).read_text());assert hashlib.sha256(Path(payload).read_bytes()).hexdigest()==meta['payload_sha256']
 with tarfile.open(payload) as archive:archive.extractall(work,filter='data')
 project=work/'project'
 curated_archive=Path('/checkpoints')/os.environ['CURATED_ARCHIVE_PATH']
 assert hashlib.sha256(curated_archive.read_bytes()).hexdigest()==os.environ['CURATED_ARCHIVE_SHA256']
 with tarfile.open(curated_archive) as archive:archive.extractall(project,filter='data')
 overlay=json.loads(zlib.decompress(base64.b64decode(os.environ['RL_SOURCE_OVERLAY'])))
 for relative,text in overlay.items():
  assert relative.startswith('src/tau2_grpo/') and '..' not in Path(relative).parts;(project/relative).write_text(text)
 run([sys.executable,'-m','compileall','-q',project/'src'])
 bank_archive=Path('/checkpoints')/os.environ['BANK_ARCHIVE_PATH']
 assert hashlib.sha256(bank_archive.read_bytes()).hexdigest()==os.environ['BANK_ARCHIVE_SHA256']
 bank=project/'short-bank';bank.mkdir()
 with tarfile.open(bank_archive) as archive:archive.extractall(bank,filter='data')
 initial=Path('/checkpoints/post-rl-clarification-sft-20261115T111954/training/run-2229ede72f/final_adapter')
 assert Path('/checkpoints/post-rl-clarification-sft-20261115T111954/SFT_COMPLETE').is_file()
 assert hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()==os.environ['INITIAL_ADAPTER_SHA256']
 for source in initial.iterdir():
  if source.is_file():shutil.copy2(source,project/'initial-adapter'/source.name)
 print('VERIFIED_SELECTED_SFT19_FOR_WEIGHTED_TRAINING',flush=True)
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['POST_RL_SFT_CONFIG']);cfg.update(model_id=base,model_revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',initial_adapter=str(project/'initial-adapter'),dataset_path=str(project/'curated-sft/train.jsonl'),eval_dataset_path=str(project/'curated-sft/validation.jsonl'),output_dir=str(store/'training'),chat_template_path=str(project/'templates/retail_sdg_v2.jinja'))
 config=project/'curated-sft-config.json';config.write_text(json.dumps(cfg,indent=2));(store/'job-config.json').write_text(json.dumps(cfg,indent=2))

 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 evaluation=json.loads(os.environ['SHORT_RL_CONFIG']);evaluation.update(scenario_bank=str(bank),model_id=base,initial_adapter=str(project/'initial-adapter'),temperature=0.,system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'))
 eval_config=project/'weighted-eval-before.json';eval_config.write_text(json.dumps(evaluation,indent=2))
 run([python,'-u','-m','tau2_grpo.rl_bank_eval','--config',eval_config,'--output',store/'short-before-weighted','--checkpoint-name','sft19-before','--worlds','80,85,86','--greedy-baseline','/checkpoints/post-rl-clarification-sft-20261115T111954/short-post-sft/summary.json'],env=env)
 # Re-run the real encoder before any optimizer update.
 run([python,'-u','-m','tau2_grpo.sft','--config',config,'--prepare-only'],env=env)
 print('POST_RL_CLARIFICATION_SFT_START',flush=True)
 run([python,'-u','-m','tau2_grpo.sft','--config',config],env=env)
 runs=[p for p in (store/'training').glob('run-*') if (p/'final_adapter/adapter_model.safetensors').exists()]
 assert len(runs)==1,runs
 selected=runs[0]/'final_adapter';state=json.loads((runs[0]/'trainer_state.json').read_text())
 result=dict(initial_checkpoint='SFT19',selected_adapter=str(selected),best_checkpoint=state.get('best_model_checkpoint'),adapter_sha256=hashlib.sha256((selected/'adapter_model.safetensors').read_bytes()).hexdigest(),training_records=500,clarification_records=200,followup_call_records=200,retention_records=100,validation_records=68,optimizer_steps=state['global_step'],complete=True)
 (store/'sft-result.json').write_text(json.dumps(result,indent=2));(store/'SFT_COMPLETE').write_text('complete\n')
 evaluation=json.loads(os.environ['SHORT_RL_CONFIG']);evaluation.update(scenario_bank=str(bank),model_id=base,initial_adapter=str(selected),temperature=0.,system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'))
 eval_config=project/'post-sft-eval.json';eval_config.write_text(json.dumps(evaluation,indent=2))
 run([python,'-u','-m','tau2_grpo.rl_bank_eval','--config',eval_config,'--output',store/'short-after-weighted','--checkpoint-name','weighted-sft-best','--worlds','80,85,86','--greedy-baseline',store/'short-before-weighted/summary.json'],env=env)
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());raise
