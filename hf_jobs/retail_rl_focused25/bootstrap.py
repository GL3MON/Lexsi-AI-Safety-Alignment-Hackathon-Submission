"""Self-contained HF Jobs bootstrap: pinned environments, Qwen, then real Tau2 RL."""
import json
import hashlib
import os
import subprocess
import sys
import tarfile
import time
import traceback
import threading
import shutil
import urllib.request
from pathlib import Path

WORK = Path('/workspace/tau2-rl')
WORK.mkdir(parents=True, exist_ok=True)
REPORTS = WORK / 'job-reports'
REPORTS.mkdir(exist_ok=True)
INPUT_REPO = os.environ['RL_INPUT_REPO']
RESULT_REPO = os.environ['RL_RESULT_REPO']
server = None
actor_server = None
report = {'stage': 'setup', 'started_at': time.time(), 'real_tau2_rewards': True}


def save():
    (REPORTS / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RL_JOB_STATUS', json.dumps(report), flush=True)


def command(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def upload_outputs():
    # Remote output uploads are disabled by default at the user's request.
    if os.environ.get('RL_UPLOAD_OUTPUTS','0')!='1':return
    from huggingface_hub import HfApi
    api = HfApi()
    # Results may include conversations and adapters; never upload environment or credentials.
    for folder, prefix in [(REPORTS, 'job-reports'), (WORK / 'project/outputs/hf-rl-smoke', 'training')]:
        if folder.exists():
            api.upload_folder(repo_id=RESULT_REPO, repo_type='dataset', folder_path=str(folder),
                              path_in_repo=prefix, commit_message='Save bounded retail RL smoke outputs',
                              ignore_patterns=['**/.netrc', '**/.env*', '**/*secrets*'])


try:
    # Install tools before importing Hub code; never dump environment variables.
    command([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'uv', 'huggingface_hub==1.33.0'])
    from huggingface_hub import hf_hub_download, snapshot_download
    payload = hf_hub_download(INPUT_REPO, 'payload.tar.gz', repo_type='dataset')
    meta=json.loads(Path(hf_hub_download(INPUT_REPO,'payload-manifest.json',repo_type='dataset')).read_text())
    if hashlib.sha256(Path(payload).read_bytes()).hexdigest()!=meta['payload_sha256']:raise RuntimeError('Payload hash mismatch')
    with tarfile.open(payload, 'r:gz') as archive:
        archive.extractall(WORK, filter='data')
    project = WORK / 'project'
    command([sys.executable,'-m','compileall','-q',project/'src'])
    initial = project / 'initial-adapter'
    if json.loads((initial / 'trainer_state.json').read_text())['global_step'] != 25:
        raise RuntimeError('Expected checkpoint 25 initialization')
    if hashlib.sha256((initial / 'adapter_model.safetensors').read_bytes()).hexdigest() != '2c72a97c0abd990dd87be650ace9674051b4e30dfff8269f1e0b8a4980b0a7f2':
        raise RuntimeError('Checkpoint 25 adapter hash mismatch')
    report['initial_checkpoint'] = 25
    config = json.loads((project / 'hf-smoke-config.json').read_text())
    command(['apt-get', 'update'], stdout=subprocess.DEVNULL)
    command(['apt-get', 'install', '-y', '--no-install-recommends',
             'git', 'build-essential', 'libnuma1', 'libgomp1'], stdout=subprocess.DEVNULL)
    teacher_env, trainer_env = WORK / 'teacher-env', WORK / 'trainer-env'
    for env in [teacher_env, trainer_env]:
        command(['uv', 'venv', '--python', sys.executable, '--seed', env])
    teacher_python = teacher_env / 'bin/python'
    trainer_python = trainer_env / 'bin/python'
    command(['uv', 'pip', 'install', '--python', teacher_python, 'vllm==0.17.1'])
    if config.get('rollout_engine') == 'vllm':
        command([teacher_python, project / 'src/tau2_grpo/vllm_compat.py'])
    command(['uv', 'pip', 'install', '--python', trainer_python,
             '--index-url', 'https://download.pytorch.org/whl/cu126', 'torch==2.11.0'])
    command(['uv', 'pip', 'install', '--python', trainer_python,
             'transformers==5.17.0', 'peft==0.19.0', 'trl==1.7.1', 'accelerate>=1.4,<2',
             'datasets==4.8.5', 'wandb==0.31.0', 'jsonschema>=4.23,<5', 'jmespath==1.0.1',
             'huggingface_hub==1.33.0', 'openai', 'scipy', 'pydub', 'soundfile', 'jiwer',
             str(WORK / 'tau2-bench') + '[gym]', str(project)])
    command(['uv', 'pip', 'install', '--python', trainer_python, '--no-deps',
             'aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git@c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'])
    hardware = subprocess.check_output([trainer_python, '-c',
        'import torch,json;f,t=torch.cuda.mem_get_info();print(json.dumps(dict(name=torch.cuda.get_device_name(),free_gib=f/2**30,total_gib=t/2**30,bf16=torch.cuda.is_bf16_supported(including_emulation=False))))'], text=True)
    report['hardware'] = json.loads(hardware)
    if not report['hardware']['bf16'] or report['hardware']['total_gib'] < 40:
        raise RuntimeError('This single-GPU layout requires BF16 and at least 40 GiB VRAM')
    save()
    base = snapshot_download('GL3MON/tiny-aya-global-function-calling-sft',
                             revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',
                             local_dir=WORK / 'base-model',
                             allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model'])
    snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model', '*.txt'])
    # vLLM 0.17.1 uses Transformers 4; our tokenizer was exported by TF5.
    # Keep the tokenizer JSON and special tokens exact; only alias the loader class.
    actor_tokenizer = WORK / 'actor-tokenizer'
    actor_tokenizer.mkdir(exist_ok=True)
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'special_tokens_map.json', 'added_tokens.json'):
        source = initial / filename
        if source.is_file(): shutil.copyfile(source, actor_tokenizer / filename)
    tokenizer_config = json.loads((actor_tokenizer / 'tokenizer_config.json').read_text())
    tokenizer_config['tokenizer_class'] = 'PreTrainedTokenizerFast'
    (actor_tokenizer / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config))
    # Check actual loader compatibility before allocating either vLLM model.
    command([teacher_python, '-c',
        'from transformers import AutoTokenizer;import json,sys;'
        't=AutoTokenizer.from_pretrained(sys.argv[1]);'
        'print(json.dumps(dict(event="actor_tokenizer_ready",eos_token_id=t.eos_token_id,vocab_size=len(t))))',
        actor_tokenizer])
    config = json.loads((project / 'hf-smoke-config.json').read_text())
    rollout_smoke = os.environ.get('RL_JOB_MODE') == 'rollout-smoke'
    if rollout_smoke:
        config.update(temperature=0.0, top_p=1.0)
        report.update(training=False, temperature=0.0, rollouts_per_task=config['group_size'])
    config.update(model_id=base, initial_adapter=str(project / 'initial-adapter'),
                  split_manifest=str(project / 'split_manifest.json'),
                  output_dir=str(project / 'outputs/hf-rl-smoke'),
                  system_prompt_file=str(project / 'templates/retail_apigen_sft_system.txt'),
                  chat_template_file=str(project / 'templates/retail_sdg_v2.jinja'),
                  user_base_url='http://127.0.0.1:8004/v1')
    (project / 'hf-runtime-config.json').write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TAU2_DATA_DIR=str(WORK / 'tau2-bench/data'),
               PYTHONPATH=str(project / 'src') + ':' + str(WORK / 'tau2-bench/src'),
               WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_ENTITY='harisshragavarchives')
    server_log = (REPORTS / 'qwen-server.log').open('w')
    server = subprocess.Popen([str(teacher_python), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', 'Qwen/Qwen3.5-4B', '--revision', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        '--served-model-name', 'qwen3.5-4b', '--host', '127.0.0.1', '--port', '8004',
        '--language-model-only', '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.28',
        '--max-model-len', '16384', '--max-num-seqs', '8', '--max-num-batched-tokens', '2048',
        '--enable-prefix-caching', '--reasoning-parser', 'qwen3', '--enforce-eager'],
        env=env, stdout=server_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('Qwen server exited; inspect qwen-server.log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8004/v1/models', timeout=5) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError('Qwen server startup exceeded ten minutes')
    if config.get('rollout_engine') == 'vllm':
        actor_log = (REPORTS / 'aya-vllm.log').open('w')
        actor_env = dict(env, VLLM_ALLOW_RUNTIME_LORA_UPDATING='True')
        actor_server = subprocess.Popen([str(teacher_python), '-m', 'vllm.entrypoints.openai.api_server',
            '--model', base, '--tokenizer', str(actor_tokenizer),
            '--served-model-name', 'aya-base', '--host', '127.0.0.1', '--port', '8005',
            '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.28',
            '--max-model-len', '8192', '--max-num-seqs', '8', '--max-num-batched-tokens', '2048',
            '--enable-lora', '--max-lora-rank', '16', '--max-loras', '1',
            '--logprobs-mode', 'processed_logprobs', '--generation-config', 'vllm',
            '--enable-prefix-caching', '--enforce-eager'],
            env=actor_env, stdout=actor_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if actor_server.poll() is not None:
                raise RuntimeError('Aya vLLM exited; inspect aya-vllm.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8005/v1/models', timeout=5) as response:
                    if response.status == 200: break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError('Aya vLLM startup exceeded ten minutes')
    # Verify the isolated dependency environment before starting an optimizer.
    command([trainer_python, '-c', 'import tau2.gym.gym_agent; from tau2_grpo.aligntune_runtime import verify_aligntune; print(verify_aligntune())'], cwd=project, env=env)
    report['stage'] = 'greedy_rollout_smoke' if rollout_smoke else 'rl_training'
    save()
    with (REPORTS / 'training.log').open('w') as log:
        trainer = subprocess.Popen([str(trainer_python), '-u', '-m', 'tau2_grpo.rl', 'collect' if rollout_smoke else 'train',
                                    '--config', str(project / 'hf-runtime-config.json')],
                                   cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        def tee_training():
            for line in trainer.stdout:
                log.write(line);log.flush()
                print(line, end='', flush=True)
        log_thread = threading.Thread(target=tee_training, daemon=True)
        log_thread.start()
        next_upload = time.monotonic() + 90
        while trainer.poll() is None:
            if os.environ.get('RL_UPLOAD_OUTPUTS','0')=='1' and time.monotonic() >= next_upload:
                upload_outputs()
                next_upload = time.monotonic() + 90
            time.sleep(5)
        log_thread.join()
        if trainer.returncode:
            raise RuntimeError(f'Trainer exited with code {trainer.returncode}; inspect training.log')
    result_files = list((project / 'outputs/hf-rl-smoke').glob('*/result.json'))
    if not result_files:
        raise RuntimeError('No training result was written')
    report.update(stage='complete', results=[json.loads(p.read_text()) for p in result_files])
    if rollout_smoke:
        report['note'] = 'Eight temperature-zero Tau2 rollout smoke episodes; no optimizer updates requested.'
    elif not any(r.get('optimizer_steps', 0) > 0 for r in report['results']):
        report['note'] = 'Rollouts completed, but all groups tied or were incomplete; no optimizer update established.'
except Exception as error:
    report.update(stage='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    raise
finally:
    report['finished_at'] = time.time()
    save()
    if server is not None and server.poll() is None:
        server.terminate()
    if actor_server is not None and actor_server.poll() is None:
        actor_server.terminate()
    try:
        upload_outputs()
    except Exception as error:
        print('OUTPUT_UPLOAD_ERROR', type(error).__name__, flush=True)
