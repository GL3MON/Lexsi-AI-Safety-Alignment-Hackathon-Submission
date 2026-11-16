"""Full mixed-curriculum GRPO, warm-started from pilot checkpoint-6."""
import os,sys,json,tarfile,subprocess,hashlib,base64,zlib,traceback
from pathlib import Path
work=Path('/workspace/midturn-full-rl');work.mkdir(parents=True,exist_ok=True);store=Path(os.environ['EVAL_OUTPUT']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kw):subprocess.run([str(a) for a in args],check=True,**kw)
try:
 archive=Path('/source')/os.environ['SOURCE_INPUT_PREFIX']/'source.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SOURCE_SHA256']
 with tarfile.open(archive) as t:t.extractall(work,filter='data')
 project=work/'project';
 for name,code in json.loads(zlib.decompress(base64.b64decode(''.join(os.environ[f'SOURCE_OVERLAY_{i}'] for i in range(int(os.environ['SOURCE_OVERLAY_PARTS'])))))).items():
  (project/'src/tau2_grpo'/name).write_text(code)
 with tarfile.open(project/'scenario-bank.tar.gz') as t:t.extractall(project/'short-bank',filter='data')
 trained=Path('/checkpoints/midturn-rl-20261115T170406/training/training/checkpoint-6/policy')
 assert (trained.parent/'COMPLETE').is_file()
 assert hashlib.sha256((trained/'adapter_model.safetensors').read_bytes()).hexdigest()==os.environ['INITIAL_ADAPTER_SHA256']
 (store/'initialization.json').write_text(json.dumps(dict(source=str(trained),sha256=os.environ['INITIAL_ADAPTER_SHA256'],optimizer='fresh AdamW',reference='frozen pilot checkpoint-6',seed=143),indent=2))
 run([sys.executable,'-m','compileall','-q',project/'src']);run([sys.executable,'-m','pip','install','-q','uv'])
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 from huggingface_hub import snapshot_download
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['EVAL_CONFIG']);cfg.update(scenario_bank=str(project/'short-bank'),model_id=base,initial_adapter=str(trained),split_manifest=str(project/'short-bank/manifest.json'),output_dir=str(work/'training'),system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'));config=project/'eval-config.json';config.write_text(json.dumps(cfg,indent=2));(store/'config.json').write_text(json.dumps(cfg,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 env.update(RL_CHECKPOINT_DIR=str(store/'training'),OMP_NUM_THREADS='4')
 # Persist device samples throughout setup, batched rollout generation and optimization.
 telemetry=open(store/'gpu.csv','w')
 monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw','--format=csv','-l','5'],stdout=telemetry,stderr=subprocess.STDOUT)
 try:
  run([python,'-u','-m','tau2_grpo.rl','train','--config',config,'--output',work/'training'],env=env)
  (store/'TRAINING_COMPLETE').write_text('complete\n')
 finally:
  monitor.terminate();monitor.wait();telemetry.close()
 (store/'EVALUATION_COMPLETE').write_text('complete\n')
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());raise
