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
 project=work/'project';overlay=json.loads(zlib.decompress(base64.b64decode(os.environ['RL_SOURCE_OVERLAY'])))
 for relative,text in overlay.items():
  assert relative.startswith('src/tau2_grpo/') and '..' not in Path(relative).parts;(project/relative).write_text(text)
 run([sys.executable,'-m','compileall','-q',project/'src'])
 bank_archive=Path('/checkpoints')/os.environ['BANK_ARCHIVE_PATH']
 assert hashlib.sha256(bank_archive.read_bytes()).hexdigest()==os.environ['BANK_ARCHIVE_SHA256']
 bank=project/'short-bank';bank.mkdir()
 with tarfile.open(bank_archive) as archive:archive.extractall(bank,filter='data')
 initial=Path('/checkpoints/post-rl-clarification-sft-20261115T111954/training/run-2229ede72f/final_adapter')
 assert Path('/checkpoints/post-rl-clarification-sft-20261115T111954/SFT_COMPLETE').is_file()
 assert hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()=='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811'
 for source in initial.iterdir():
  if source.is_file():shutil.copy2(source,project/'initial-adapter'/source.name)
 print('VERIFIED_SHORT_BANK_AND_SELECTED_SFT19',flush=True)
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['SHORT_RL_CONFIG']);cfg.update(scenario_bank=str(bank),model_id=base,initial_adapter=str(project/'initial-adapter'),split_manifest=str(project/'split_manifest.json'),output_dir=str(project/'outputs/short-retail-rl'),system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'))
 config=project/'short-rl-config.json';config.write_text(json.dumps(cfg,indent=2));(store/'job-config.json').write_text(json.dumps(cfg,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 run([python,'-m','tau2_grpo.rl','check','--config',config],env=env)
 print('SHORT_RL_START',json.dumps(dict(groups=cfg['train_steps'],rollouts_per_group=cfg['group_size'],max_actions=cfg['max_actions'],reward='short-verifiable, not native Tau2',before_after_greedy_cases=30)),flush=True)
 run([python,'-u','-m','tau2_grpo.rl','train','--config',config],env=env)
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());raise
