"""Inference-only90-case retail check of the diagnostic SFT50 checkpoint."""
import os,sys,json,tarfile,subprocess,hashlib,base64,zlib,traceback
from pathlib import Path
work=Path('/workspace/sft50-retail90');work.mkdir(parents=True,exist_ok=True);store=Path(os.environ['EVAL_OUTPUT']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kw):subprocess.run([str(a) for a in args],check=True,**kw)
try:
 archive=Path('/source')/os.environ['SOURCE_INPUT_PREFIX']/'source.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SOURCE_SHA256']
 with tarfile.open(archive) as t:t.extractall(work,filter='data')
 project=work/'project';(project/'src/tau2_grpo/rl_bank_eval.py').write_text(zlib.decompress(base64.b64decode(os.environ['EVALUATOR_CODE'])).decode())
 with tarfile.open(project/'scenario-bank.tar.gz') as t:t.extractall(project/'short-bank',filter='data')
 trained=Path('/checkpoints/sft-decision-diagnostic-20261115T152936/training/run-b2b447c0a9/final_adapter');assert Path('/checkpoints/sft-decision-diagnostic-20261115T152936/SFT_COMPLETE').is_file()
 assert hashlib.sha256((trained/'adapter_model.safetensors').read_bytes()).hexdigest()=='eae5e7bc4d3d2338ac3afaa2ec9a549808a6fa57f17844f3e95d4feea11a6e1d'
 run([sys.executable,'-m','compileall','-q',project/'src']);run([sys.executable,'-m','pip','install','-q','uv'])
 run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project])
 from huggingface_hub import snapshot_download
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['EVAL_CONFIG']);cfg.update(scenario_bank=str(project/'short-bank'),model_id=base,initial_adapter=str(trained),temperature=0.,system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'));config=project/'eval-config.json';config.write_text(json.dumps(cfg,indent=2));(store/'config.json').write_text(json.dumps(cfg,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false')
 run([python,'-u','-m','tau2_grpo.rl_bank_eval','--config',config,'--output',store/'short-retail90','--checkpoint-name','diagnostic-sft50','--worlds','80,85,86','--greedy-baseline',project/'baseline-summary.json'],env=env)
 (store/'EVALUATION_COMPLETE').write_text('complete\n')
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());raise
