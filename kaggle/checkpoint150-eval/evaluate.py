"""Private Kaggle checkpoint-150 inference diagnostic. No training or credentials."""
import hashlib,json,os,subprocess,sys,traceback,zipfile
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
out=Path('/kaggle/working/checkpoint150-tool-eval');out.mkdir(parents=True,exist_ok=True)
try:
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.17.0','peft==0.19.0','accelerate'],check=True)
    # Kaggle automatically extracts dataset ZIPs; support both mounted forms.
    bundle=next((p.parent for p in Path('/kaggle/input').rglob('manifest.json')
                 if (p.parent/'checkpoint/adapter_model.safetensors').is_file()),None)
    if bundle is None:
        archive=next(iter(Path('/kaggle/input').rglob('*eval.zip')),None)
        if archive is None:raise RuntimeError('Checkpoint input is missing: no extracted checkpoint folder or eval ZIP was mounted')
        bundle=Path('/kaggle/working/eval-input');bundle.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:z.extractall(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert hashlib.sha256((bundle/'checkpoint/adapter_model.safetensors').read_bytes()).hexdigest()==manifest['adapter_sha256']
    print('CHECKPOINT_MANIFEST',json.dumps(manifest),flush=True)
    subprocess.run([sys.executable,'-u',str(bundle/'evaluate.py'),'--checkpoint',str(bundle/'checkpoint'),
        '--system',str(bundle/'retail_system_prompt.txt'),'--output',str(out)],check=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps(dict(error=str(error),traceback=traceback.format_exc()),indent=2))
    raise
