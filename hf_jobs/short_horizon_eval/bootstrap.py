import os,sys,json,tarfile,subprocess,zlib,base64
from pathlib import Path
from huggingface_hub import hf_hub_download
work=Path('/workspace/short-horizon');work.mkdir(parents=True,exist_ok=True)
def run(args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
run([sys.executable,'-m','pip','install','-q','uv'])
payload=hf_hub_download('GL3MON/tau2-retail-focused25-rl-input','payload.tar.gz',repo_type='dataset')
with tarfile.open(payload) as archive:archive.extractall(work,filter='data')
run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','accelerate>=1.4,<2','huggingface_hub==1.33.0',work/'tau2-bench'])
script=work/'evaluate.py';script.write_bytes(zlib.decompress(base64.b64decode(os.environ['SHORT_EVAL_CODE'])))
env=dict(os.environ,SHORT_WORK=str(work),PYTHONPATH=str(work/'tau2-bench/src'),TAU2_DATA_DIR=str(work/'tau2-bench/data'),CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false')
run([python,'-u',script],env=env)
