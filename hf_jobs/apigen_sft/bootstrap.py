"""Single-GPU Salesforce retail SFT: measured batch size, one epoch, online W&B."""
import hashlib,json,math,os,subprocess,sys,tarfile,threading,time,traceback
from pathlib import Path

WORK=Path('/workspace/apigen-sft');WORK.mkdir(parents=True,exist_ok=True)
REPORTS=WORK/'reports';REPORTS.mkdir(exist_ok=True)
status=dict(stage='setup',epochs=1,initial_checkpoint='glaive-rl17')

def save():
    (REPORTS/'status.json').write_text(json.dumps(status,indent=2))
    print('SFT_JOB_STATUS',json.dumps(status),flush=True)

def run(args,**kwargs):subprocess.run([str(x) for x in args],check=True,**kwargs)

def upload():
    from huggingface_hub import HfApi
    api=HfApi()
    for path,prefix in [(REPORTS,'reports'),(WORK/'project/outputs/sft-apigen-retail','training')]:
        if path.exists():
            api.upload_folder(repo_id=os.environ['SFT_RESULT_REPO'],repo_type='dataset',folder_path=str(path),path_in_repo=prefix,
                ignore_patterns=['**/wandb/**','**/.env*','**/.netrc'])

def main():
    run([sys.executable,'-m','pip','install','-q','uv','huggingface_hub==1.33.0'])
    from huggingface_hub import hf_hub_download,snapshot_download
    payload=hf_hub_download(os.environ['SFT_INPUT_REPO'],'payload.tar.gz',repo_type='dataset')
    meta=json.loads(Path(hf_hub_download(os.environ['SFT_INPUT_REPO'],'manifest.json',repo_type='dataset')).read_text())
    assert hashlib.sha256(Path(payload).read_bytes()).hexdigest()==meta['payload_sha256']
    with tarfile.open(payload) as archive:archive.extractall(WORK,filter='data')
    project=WORK/'project'
    assert hashlib.sha256((project/'initial-adapter/adapter_model.safetensors').read_bytes()).hexdigest()==meta['initial_adapter_sha256']
    run(['apt-get','update'],stdout=subprocess.DEVNULL)
    run(['apt-get','install','-y','--no-install-recommends','git','build-essential','libgomp1'],stdout=subprocess.DEVNULL)
    venv=WORK/'env';run(['uv','venv',str(venv),'--python',sys.executable]);python=venv/'bin/python'
    run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.11.0'])
    run(['uv','pip','install','--python',python,'transformers==5.17.0','peft==0.19.0','trl==1.7.1',
         'accelerate>=1.4,<2','datasets==4.8.5','wandb==0.31.0','huggingface_hub==1.33.0','jsonschema','jmespath',str(project)])
    run(['uv','pip','install','--python',python,'--no-deps',
         'aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(project/'src'),TOKENIZERS_PARALLELISM='false',
             WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_ENTITY='harisshragavarchives')
    hardware=json.loads(subprocess.check_output([python,'-c','import torch,json;print(json.dumps(dict(name=torch.cuda.get_device_name(),total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,bf16=torch.cuda.is_bf16_supported())))'],text=True,env=env))
    assert hardware['total_gib']>=40 and hardware['bf16'];status['hardware']=hardware;save()
    base=snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',
                           local_dir=WORK/'base',allow_patterns=['*.json','*.safetensors','*.jinja','*.model'])
    config=json.loads((project/'sft-config.json').read_text())
    config.update(model_id=base,model_revision='main',initial_adapter=str(project/'initial-adapter'),
        dataset_path=str(project/'data/train.jsonl'),eval_dataset_path=str(project/'data/validation.jsonl'),
        chat_template_path=str(project/'templates/retail_sdg_v2.jinja'),output_dir=str(project/'outputs/sft-apigen-retail'))
    cfg_file=project/'runtime-config.json';cfg_file.write_text(json.dumps(config,indent=2))
    reuse=meta.get('validated_batch_probe')
    if reuse:
        assert reuse['selected']['batch_size']==config['batch_size']==8
        assert reuse['max_context_tokens']>=meta['dataset']['max_tokens']
        assert reuse['selected']['total_gpu_gib']<=hardware['total_gib']+.1
        print('REUSING_MEASURED_BATCH',json.dumps(reuse),flush=True)
    status['stage']='batch_probe';save();attempts=[reuse['selected']] if reuse else []
    for batch in ([] if reuse else [8,16,24,32]):
        target=REPORTS/f'batch-{batch}.json'
        print('PROBING_BATCH',batch,flush=True)
        with (REPORTS/f'batch-{batch}.log').open('w') as log:
            run([python,project/'probe.py','--worker','--config',cfg_file,'--sample-file',project/'data/longest-tokenized.json',
                 '--batch-size',batch,'--result-path',target],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(target.read_text());attempts.append(result);print('BATCH_RESULT',json.dumps(result),flush=True)
        if result['status'] in ('oom','allocator_error'):break
        if result['status']!='ok' or not math.isfinite(result['train_loss']):raise RuntimeError('Batch probe failed: '+str(result))
        if result['peak_reserved_gib']>hardware['total_gib']-2:break
    usable=[r for r in attempts if r['status']=='ok' and r['peak_reserved_gib']<hardware['total_gib']-2]
    if not usable:
        raise RuntimeError('No safe batch >=8; inspect probe before retrying at a smaller batch')
    best=max(usable,key=lambda r:r['batch_size']/r['step_time_seconds'])
    config['batch_size']=best['batch_size'];cfg_file.write_text(json.dumps(config,indent=2))
    status.update(stage='training',batch_size=best['batch_size'],gradient_accumulation_steps=config['gradient_accumulation_steps'],
                  effective_batch_size=best['batch_size']*config['gradient_accumulation_steps'],
                  estimated_optimizer_steps=math.ceil(math.ceil(meta['dataset']['train']/best['batch_size'])/config['gradient_accumulation_steps']))
    (REPORTS/'batch-summary.json').write_text(json.dumps(dict(selected=best,attempts=attempts),indent=2));save();upload()
    with (REPORTS/'training.log').open('w') as logfile:
        process=subprocess.Popen([str(python),'-u','-m','tau2_grpo.sft','--config',str(cfg_file)],cwd=project,env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def tee():
            for line in process.stdout:logfile.write(line);logfile.flush();print(line,end='',flush=True)
        thread=threading.Thread(target=tee,daemon=True);thread.start();next_upload=time.monotonic()+120
        while process.poll() is None:
            if time.monotonic()>=next_upload:
                try:upload()
                except Exception as error:print('PERIODIC_UPLOAD_ERROR',type(error).__name__,flush=True)
                next_upload=time.monotonic()+120
            time.sleep(5)
        thread.join()
        if process.returncode:raise RuntimeError('Training process exited '+str(process.returncode))
    metrics=list((project/'outputs/sft-apigen-retail').glob('*/metrics.json'))
    if not metrics:raise RuntimeError('No final SFT metrics were written')
    status.update(stage='complete',metrics=json.loads(metrics[-1].read_text()));save()

if __name__=='__main__':
    try:main()
    except Exception as error:
        status.update(stage='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:
        save()
        try:upload()
        except Exception as error:print('FINAL_UPLOAD_ERROR',str(error),flush=True)
