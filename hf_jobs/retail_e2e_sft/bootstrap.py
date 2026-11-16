"""Verified full-trajectory SFT, one unassisted rollout, then exit."""
import os,sys,json,tarfile,hashlib,subprocess,base64,zlib,traceback
from pathlib import Path
work=Path('/workspace/retail-e2e-sft');work.mkdir(parents=True,exist_ok=True);store=Path(os.environ['EVAL_OUTPUT']);store.mkdir(parents=True,exist_ok=True)
def run(args,**kw):subprocess.run([str(x) for x in args],check=True,**kw)
try:
 archive=Path('/source')/os.environ['SOURCE_INPUT_PREFIX']/'source.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SOURCE_SHA256']
 with tarfile.open(archive) as t:t.extractall(work,filter='data')
 project=work/'project'
 for name,code in json.loads(zlib.decompress(base64.b64decode(''.join(os.environ[f'SOURCE_OVERLAY_{i}'] for i in range(int(os.environ['SOURCE_OVERLAY_PARTS'])))))).items():(project/'src/tau2_grpo'/name).write_text(code)
 archive=Path(os.environ['SFT_DATA_ARCHIVE']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SFT_DATA_SHA256']
 with tarfile.open(archive) as t:t.extractall(project/'e2e-data',filter='data')
 initial=Path('/checkpoints/midturn-full-rl-20261115T173756/training/training/checkpoint-7/policy');assert (initial.parent/'COMPLETE').is_file();assert hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()==os.environ['INITIAL_ADAPTER_SHA256']
 data=project/'e2e-data';manifest=json.loads((data/'manifest.json').read_text())
 for name,sha in manifest['data_sha256'].items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha
 run([sys.executable,'-m','pip','install','-q','uv']);run(['apt-get','update'],stdout=subprocess.DEVNULL);run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
 venv=work/'env';run(['uv','venv',venv,'--python',sys.executable]);python=venv/'bin/python'
 run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
 run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1','accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath','openai','scipy','pydub','soundfile','jiwer',str(work/'tau2-bench')+'[gym]',project]);run(['uv','pip','install','--python',python,'--no-deps','aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
 from huggingface_hub import snapshot_download
 base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
 cfg=json.loads(os.environ['SFT_CONFIG']);cfg.update(model_id=base,initial_adapter=str(initial),dataset_path=str(data/'train.jsonl'),eval_dataset_path=str(data/'validation.jsonl'),task_split_manifest=str(data/'task-split-manifest.json'),chat_template_path=str(project/'templates/retail_sdg_v2.jinja'),output_dir=str(store/'training'));config=project/'sft-config.json';config.write_text(json.dumps(cfg,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TAU2_DATA_DIR=str(work/'tau2-bench/data'),PYTHONPATH=str(project/'src')+':'+str(work/'tau2-bench/src'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 probe_attempts=[]
 for batch,accum in [(8,2),(4,4),(2,8)]:
  result=store/f'probe-batch-{batch}.json';run([python,data/'probe.py','--worker','--config',config,'--sample-file',data/'longest-tokenized.json','--batch-size',batch,'--result-path',result],env=env);probe=json.loads(result.read_text());probe_attempts.append(probe)
  if probe['status']=='ok' and probe['peak_reserved_gib']<probe['total_gpu_gib']-1.5:cfg.update(batch_size=batch,gradient_accumulation_steps=accum);break
  if probe['status'] not in ('oom','allocator_error','ok'):raise RuntimeError('Batch probe failed: '+str(probe))
 else:raise RuntimeError('No tested batch size fits with memory margin')
 config.write_text(json.dumps(cfg,indent=2));(store/'config.json').write_text(json.dumps(cfg,indent=2));(store/'batch-probe.json').write_text(json.dumps(dict(selected_batch_size=cfg['batch_size'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],attempts=probe_attempts),indent=2))
 run([python,'-u','-m','tau2_grpo.sft','--config',config],env=env)
 runs=[p for p in (store/'training').glob('run-*') if (p/'final_adapter/adapter_model.safetensors').exists()];assert len(runs)==1;trained=runs[0]/'final_adapter';state=json.loads((runs[0]/'trainer_state.json').read_text());(store/'selection.json').write_text(json.dumps(dict(final_adapter=str(trained),global_step=state['global_step'],best_model_checkpoint=state.get('best_model_checkpoint'),best_metric=state.get('best_metric'),sha256=hashlib.sha256((trained/'adapter_model.safetensors').read_bytes()).hexdigest()),indent=2));(store/'SFT_COMPLETE').write_text('complete\n')
 evaluation=json.loads(os.environ['ROLLOUT_CONFIG']);evaluation.update(model_id=base,initial_adapter=str(trained),environment='tau2',assistance_level='none',assisted_adaptive=False,split_manifest=str(project/'split_manifest.json'),system_prompt_file=str(project/'templates/retail_apigen_sft_system.txt'),chat_template_file=str(project/'templates/retail_sdg_v2.jinja'));(project/'split_manifest.json').write_text(os.environ['TAU2_SPLIT_MANIFEST']);ec=project/'rollout-config.json';ec.write_text(json.dumps(evaluation,indent=2))
 run([python,'-u','-m','tau2_grpo.e2e_sft_rollout','--config',ec,'--output',store/'tau2-test','--task-id','69'],env=env)
 (store/'JOB_FINISHED').write_text('SFT and one rollout test finished. No further jobs scheduled.\n')
except Exception:
 (store/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True);(store/'FAILED').write_text('See error.txt; no automatic retry or follow-on training.\n')
