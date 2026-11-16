import os,sys,json,subprocess,tarfile,traceback,shutil
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
work=Path('/workspace/clarification');work.mkdir(parents=True,exist_ok=True)
out=work/'results';out.mkdir(exist_ok=True);api=HfApi()
def run(args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
def upload():api.upload_folder(repo_id=os.environ['EVAL_RESULTS'],repo_type='dataset',folder_path=str(out))
try:
    run([sys.executable,'-m','pip','install','-q','uv'])
    run(['apt-get','update'],stdout=subprocess.DEVNULL)
    run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
    archive=hf_hub_download('GL3MON/tiny-aya-apigen-retail-sft-input','payload.tar.gz',repo_type='dataset')
    with tarfile.open(archive) as tar:tar.extractall(work,filter='data')
    project=work/'project';venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
    run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
    run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath',project])
    run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
    source=os.environ['EVAL_INPUT']
    for name in ['train.jsonl','validation.jsonl','manifest.json']:
        shutil.copy(hf_hub_download(source,name,repo_type='dataset'),project/'data'/name)
    prefix='training/run-2a32064432/checkpoint-50'
    root=snapshot_download('GL3MON/tiny-aya-apigen-retail-sft-results',repo_type='dataset',allow_patterns=[prefix+'/*'],ignore_patterns=['*/optimizer.pt','*/scheduler.pt','*/rng_state.pth'])
    cfg=json.loads((project/'sft-config.json').read_text())
    cfg.update(initial_adapter=str(Path(root)/prefix),dataset_path=str(project/'data/train.jsonl'),eval_dataset_path=str(project/'data/validation.jsonl'),output_dir=str(out/'training'),chat_template_path=str(project/'templates/retail_sdg_v2.jinja'),batch_size=8,eval_batch_size=2,gradient_accumulation_steps=1,num_train_epochs=1.,learning_rate=2e-5,eval_steps=5,save_steps=5,save_total_limit=2,evaluate_before_training=True,preprocessing_num_workers=4,wandb_run_name='targeted-clarification-200-from-apigen50',wandb_tags=['targeted-clarification','200-samples','100-missing-100-complete','from-apigen50'],log_sample_count=0)
    config=project/'clarification-config.json';config.write_text(json.dumps(cfg,indent=2));shutil.copy(config,out/'config.json')
    env=dict(os.environ,PYTHONPATH=str(project/'src'),CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false')
    # Verify all labels and 8K bounds using the same training encoder before any update.
    audit='''import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records,encode_records
cfg=json.load(open(sys.argv[1]));tok=AutoTokenizer.from_pretrained(cfg['initial_adapter']);tok.chat_template=Path(cfg['chat_template_path']).read_text();n=0;max_tokens=0
for row,enc,spans,error in encode_records(tok,read_records(Path(cfg['dataset_path']))+read_records(Path(cfg['eval_dataset_path'])),8192,1):
 assert not error,(row['id'],error)
 indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant'];assert set(indices)==set(row['loss_message_indices']);selected=set()
 for start,end in spans:
  assert enc['labels'][start:end]==enc['input_ids'][start:end];selected.update(range(start,end))
 assert all(v==-100 for i,v in enumerate(enc['labels']) if i not in selected)
 n+=1;max_tokens=max(max_tokens,len(enc['input_ids']))
print(json.dumps(dict(records=n,max_tokens=max_tokens,assistant_only_masks_verified=True)))
'''
    (project/'audit.py').write_text(audit)
    run([python,project/'audit.py',config],env=env)
    with (out/'training.log').open('w') as log:
        proc=subprocess.Popen([str(python),'-u','-m','tau2_grpo.sft','--config',str(config)],cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        if proc.wait():raise RuntimeError('SFT failed')
    upload()
    run_dir=next((out/'training').glob('run-*'))
    state=json.loads((run_dir/'trainer_state.json').read_text())
    best=state.get('best_model_checkpoint');checkpoint=Path(best) if best else run_dir/'final_adapter'
    script=hf_hub_download(source,'evaluate.py',repo_type='dataset');system=hf_hub_download(source,'system.txt',repo_type='dataset')
    run([python,script,'--checkpoint',checkpoint,'--system',system,'--output',out/'evaluation'],env=env)
    (out/'experiment.json').write_text(json.dumps({'initial_checkpoint':50,'training_samples':200,'missing_information':100,'complete_information':100,'validation_samples':24,'eval_cases':24,'baseline_reference':'GL3MON/aya-clear-clarification-eval-20261114T174418/checkpoint-50','baseline_score':8,'baseline_total':24,'grading_note':'Raw lexical clarification checks require review for invented results and requests for already-known fields.'},indent=2))
    upload()
except Exception:
    (out/'error.txt').write_text(traceback.format_exc());upload();raise
