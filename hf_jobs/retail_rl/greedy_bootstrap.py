"""Bounded inference-only checkpoint-100 greedy diagnostic; never trains."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv', 'huggingface_hub==1.33.0'], check=True)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    root = Path('/workspace/greedy-check');root.mkdir(parents=True, exist_ok=True)
    out = root/'reports';out.mkdir(exist_ok=True)
    api = HfApi()
    try:
        subprocess.run(['apt-get','update'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['apt-get','install','-y','--no-install-recommends',
                        'build-essential','libnuma1','libgomp1'],check=True,stdout=subprocess.DEVNULL)
        payload = hf_hub_download(os.environ['RL_INPUT_REPO'], 'payload.tar.gz', repo_type='dataset')
        with tarfile.open(payload) as archive:archive.extractall(root, filter='data')
        initial = root/'project/initial-adapter'
        assert json.loads((initial/'trainer_state.json').read_text())['global_step'] == 100
        assert hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest() == '97bf4daa0de4fcc48a3ae31c6cd60f77d916fcb4d6ada11cff8191fbcfc9e176'
        prompt = hf_hub_download(os.environ['RL_INPUT_REPO'], 'greedy-prompt.json', repo_type='dataset')
        base = snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',
            revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a', local_dir=root/'base',
            allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
        tokenizer = root/'tokenizer';tokenizer.mkdir(exist_ok=True)
        for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
            shutil.copyfile(initial/name, tokenizer/name)
        config = json.loads((tokenizer/'tokenizer_config.json').read_text())
        config['tokenizer_class'] = 'PreTrainedTokenizerFast'
        (tokenizer/'tokenizer_config.json').write_text(json.dumps(config))
        subprocess.run(['uv','venv',str(root/'env'),'--python',sys.executable],check=True)
        python = root/'env/bin/python'
        subprocess.run(['uv','pip','install','--python',str(python),'vllm==0.17.1'],check=True)
        subprocess.run([python,root/'project/src/tau2_grpo/vllm_compat.py'],check=True)
        # Run in a clean interpreter after dependency installation and patching.
        script = r'''
import json,sys
from pathlib import Path
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
base,tokenizer,initial,prompt,out=sys.argv[1:]
out=Path(out);t=AutoTokenizer.from_pretrained(tokenizer)
source=json.loads(Path(prompt).read_text())
assert t.encode(source['rendered_prompt'],add_special_tokens=False)==source['prompt_ids']
llm=LLM(model=base,tokenizer=tokenizer,dtype='bfloat16',max_model_len=8192,
    gpu_memory_utilization=.28,max_num_seqs=8,max_num_batched_tokens=2048,
    enable_lora=True,max_lora_rank=16,max_loras=1,enforce_eager=True,
    logprobs_mode='processed_logprobs',generation_config='vllm',enable_prefix_caching=True)
params=SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=128,
    ignore_eos=True,stop_token_ids=[6],skip_special_tokens=False,seed=42)
adapter=LoRARequest('checkpoint100',1,initial)
short=t.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
    {'role':'user','content':'Hello. Please introduce yourself briefly.'}],tokenize=True,add_generation_prompt=True)
results=[]
for label,ids,lora,count in [('base-long',source['prompt_ids'],None,1),
    ('checkpoint100-long',source['prompt_ids'],adapter,1),
    ('checkpoint100-short',short,adapter,1),
    ('checkpoint100-long-batch8',source['prompt_ids'],adapter,8)]:
    generated=llm.generate([{'prompt_token_ids':ids} for _ in range(count)],params,lora_request=lora,use_tqdm=False)
    for index,response in enumerate(generated):
        output=response.outputs[0]
        row=dict(case=label,index=index,temperature=0,checkpoint=100 if lora else None,
            prompt_tokens=len(ids),completion_ids=list(output.token_ids),
            text=t.decode(output.token_ids,skip_special_tokens=True),finish_reason=output.finish_reason)
        results.append(row);print(json.dumps(row),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'conversations.txt').write_text('\n\n'.join(
        f"=== {r['case']} / {r['index']} | temperature=0 ===\nASSISTANT:\n{r['text']}" for r in results))
'''
        subprocess.run([python,'-u','-c',script,base,str(tokenizer),str(initial),prompt,str(out)],check=True)
        (out/'status.json').write_text(json.dumps({'complete':True,'training':False,'temperature':0}))
    except Exception as error:
        (out/'status.json').write_text(json.dumps({'complete':False,'training':False,'error':str(error)}))
        raise
    finally:
        api.upload_folder(repo_id=os.environ['RL_RESULT_REPO'],repo_type='dataset',
            folder_path=str(out),path_in_repo='greedy-checkpoint100',commit_message='Checkpoint 100 greedy inference diagnostics')


if __name__=='__main__':main()
