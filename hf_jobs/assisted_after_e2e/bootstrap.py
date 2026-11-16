"""Assisted native Tau2 curriculum: read-only context, sampled suffix targets."""
import os,sys,json,tarfile,subprocess,hashlib,base64,zlib,traceback
from pathlib import Path
work=Path('/workspace/tau2-assisted-rl');work.mkdir(parents=True,exist_ok=True);store=Path(os.environ['EVAL_OUTPUT']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kw):subprocess.run([str(a) for a in args],check=True,**kw)
try:
 archive=Path('/source')/os.environ['SOURCE_INPUT_PREFIX']/'source.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SOURCE_SHA256']
 with tarfile.open(archive) as t:t.extractall(work,filter='data')
 project=work/'project';
 for name,code in json.loads(zlib.decompress(base64.b64decode(''.join(os.environ[f'SOURCE_OVERLAY_{i}'] for i in range(int(os.environ['SOURCE_OVERLAY_PARTS'])))))).items():
  (project/'src/tau2_grpo'/name).write_text(code)
 with tarfile.open(project/'scenario-bank.tar.gz') as t:t.extractall(project/'short-bank',filter='data')
 trained=Path(os.environ['INITIAL_ADAPTER_PATH'])
 if os.environ.get('INITIAL_CHECKPOINT_STEP'):
  state=json.loads((trained/'trainer_state.json').read_text());assert state['global_step']==int(os.environ['INITIAL_CHECKPOINT_STEP'])
  assert (trained/'adapter_config.json').is_file()
 else:assert Path(os.environ['SFT_COMPLETE_MARKER']).is_file()
 assert hashlib.sha256((trained/'adapter_model.safetensors').read_bytes()).hexdigest()==os.environ['INITIAL_ADAPTER_SHA256']
 (store/'initialization.json').write_text(json.dumps(dict(source=str(trained),sha256=os.environ['INITIAL_ADAPTER_SHA256'],optimizer='fresh AdamW',reference='frozen selected end-to-end SFT checkpoint',seed=345),indent=2))
 run([sys.executable,'-m','compileall','-q',project/'src']);run([sys.executable,'-m','pip','install','-q','uv'])
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','fastapi','uvicorn','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 from huggingface_hub import snapshot_download
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['EVAL_CONFIG']);cfg.update(scenario_bank=str(project/'short-bank'),model_id=base,initial_adapter=str(trained),split_manifest=str(project/'split_manifest.json'),curriculum_file=str(project/'tau2-curriculum.json'),output_dir=str(work/'training'),system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'));config=project/'eval-config.json';config.write_text(json.dumps(cfg,indent=2));(store/'config.json').write_text(json.dumps(cfg,indent=2))
 (project/'split_manifest.json').write_text(os.environ['TAU2_SPLIT_MANIFEST'])
 (project/'tau2-curriculum.json').write_text(os.environ['TAU2_CURRICULUM'])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 env.update(RL_CHECKPOINT_DIR=str(store/'training'),OMP_NUM_THREADS='4')
 # Persist device samples throughout setup, batched rollout generation and optimization.
 telemetry=open(store/'gpu.csv','w')
 monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw','--format=csv','-l','5'],stdout=telemetry,stderr=subprocess.STDOUT)
 server_log=open(store/'qwen-server.log','w')
 server=subprocess.Popen([str(python),'-u','-m','tau2_grpo.frozen_qwen_server','--config',str(config)],env=env,stdout=server_log,stderr=subprocess.STDOUT)
 try:
  import time,urllib.request
  deadline=time.monotonic()+600
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError('Frozen Qwen exited; see qwen-server.log')
   try:
    with urllib.request.urlopen('http://127.0.0.1:8004/v1/models',timeout=3) as response:
     if response.status==200:break
   except Exception:time.sleep(3)
  else:raise RuntimeError('Frozen Qwen startup timeout')
  if os.environ.get('RESUME_SMOKE_PREFIX'):
   import shutil
   previous=Path('/checkpoints')/os.environ['RESUME_SMOKE_PREFIX']
   previous_init=json.loads((previous/'initialization.json').read_text());previous_cfg=json.loads((previous/'config.json').read_text())
   if previous_init['sha256']==os.environ['INITIAL_ADAPTER_SHA256'] and previous_cfg==cfg:
    destination=store/'smoke/group-0000';destination.mkdir(parents=True,exist_ok=True)
    reused=0
    for file in (previous/'smoke/group-0000').glob('episode-*.json'):
     episode=json.loads(file.read_text())
     if episode.get('complete') and episode.get('reward') is not None:
      shutil.copyfile(file,destination/file.name);reused+=1
    (store/'smoke/CACHE_VERIFIED').write_text(json.dumps(dict(source=str(previous),sha256=previous_init['sha256'],reused=reused)))
    print('VERIFIED_SMOKE_REUSE',reused,flush=True)
  run([python,'-u','-m','tau2_grpo.tau2_smoke','--config',config,'--output',store/'smoke'],env=env)
  run([python,'-u','-m','tau2_grpo.rl','train','--config',config,'--output',work/'training'],env=env)
  (store/'TRAINING_COMPLETE').write_text('complete\n')
 finally:
  server.terminate();server.wait();server_log.close()
  monitor.terminate();monitor.wait();telemetry.close()
 (store/'EVALUATION_COMPLETE').write_text('complete\n')
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True);(store/'FAILED').write_text('Training did not complete; see error.txt\n')
