"""Portable SFT decision diagnostic: all private inputs belong to the newly authenticated account."""
import os,sys,json,subprocess,tarfile,hashlib,shutil,traceback,base64,zlib
from pathlib import Path
work=Path('/workspace/clarification-diagnostic');work.mkdir(parents=True,exist_ok=True)
store=Path(os.environ['DIAG_OUTPUT_DIR']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
try:
 archive=Path('/checkpoints')/os.environ['DIAG_INPUT_PREFIX']/'source.tar.gz'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['DIAG_INPUT_SHA256']
 with tarfile.open(archive) as tar:tar.extractall(work,filter='data')
 project=work/'project'
 (project/'src/tau2_grpo/sft_diagnostic.py').write_text(zlib.decompress(base64.b64decode(os.environ['FIXED_EVALUATOR'])).decode())
 initial=project/'initial-adapter';initial.mkdir()
 for name,sha in json.loads(os.environ['DIAG_INITIAL_FILES']).items():
  source=Path('/checkpoints')/os.environ['DIAG_INPUT_PREFIX']/'initial-adapter'/name
  assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
  shutil.copy2(source,initial/name)
 assert hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()=='98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811'
 run([sys.executable,'-m','compileall','-q',project/'src'])
 print('PORTABLE_SFT_DIAGNOSTIC_VERIFIED',flush=True)
 run([sys.executable,'-m','pip','install','-q','uv'])
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 from huggingface_hub import snapshot_download
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 evaluation=json.loads(os.environ['DIAG_EVAL_CONFIG']);evaluation.update(model_id=base,initial_adapter=str(initial),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'),temperature=0.)
 eval_config=project/'eval-config.json';eval_config.write_text(json.dumps(evaluation,indent=2))
 run([python,'-u','-m','tau2_grpo.sft_diagnostic','--config',eval_config,'--data',project/'diagnostic','--output',store/'evaluation','--phase','before'],env=env)
 trained=Path('/checkpoints/sft-decision-diagnostic-20261115T152936/training/run-b2b447c0a9/final_adapter')
 assert Path('/checkpoints/sft-decision-diagnostic-20261115T152936/SFT_COMPLETE').is_file()
 assert hashlib.sha256((trained/'adapter_model.safetensors').read_bytes()).hexdigest()=='eae5e7bc4d3d2338ac3afaa2ec9a549808a6fa57f17844f3e95d4feea11a6e1d'
 evaluation['initial_adapter']=str(trained);eval_config.write_text(json.dumps(evaluation,indent=2))
 run([python,'-u','-m','tau2_grpo.sft_diagnostic','--config',eval_config,'--data',project/'diagnostic','--output',store/'evaluation','--phase','after'],env=env)
 (store/'EVALUATION_COMPLETE').write_text('complete\n')
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());raise
