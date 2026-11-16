"""Isolated Tau2 processes with batched inference on one parent Transformers model."""
from concurrent.futures import TimeoutError
from dataclasses import asdict
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import time


class WorkerPolicy:
    def __init__(self, connection, tokenizer, version):
        self.connection, self.tokenizer, self.version = connection, tokenizer, version

    def generate(self, prepared, max_tokens, seed):
        self.connection.send(('generate', prepared, max_tokens, seed))
        kind, result = self.connection.recv()
        if kind != 'result': raise RuntimeError(result)
        return result


def worker(connection, raw_config, version, task_id, seed, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from loguru import logger
    logger.remove();logger.add(__import__('sys').stderr, level='ERROR')
    from transformers import AutoTokenizer
    from tau2_grpo.rl import make_episode
    from tau2_grpo.rl_config import RetailRLConfig, path
    try:
        cfg = RetailRLConfig(**raw_config)
        tokenizer = AutoTokenizer.from_pretrained(str(path(cfg.initial_adapter)))
        if cfg.chat_template_file:tokenizer.chat_template = path(cfg.chat_template_file).read_text()
        make_episode(WorkerPolicy(connection, tokenizer, version), cfg, task_id, seed, output)
        connection.send(('done', output))
    except Exception as error:
        connection.send(('error', f'{type(error).__name__}: {error}'))
    finally:
        connection.close()


def serve_requests(policy, requests):
    """Keep each request's output reserve; never silently shorten another row."""
    buckets = {}
    for connection, prepared, max_tokens, seed in requests:
        buckets.setdefault(max_tokens, []).append((connection, prepared, seed))
    for max_tokens, batch in buckets.items():
        try:
            results = policy.generate_batch([item[1] for item in batch], max_tokens,
                                            seed=min(item[2] for item in batch))
            if len(results) != len(batch):raise RuntimeError('Unaligned generation batch')
            for (connection, _, _), result in zip(batch, results, strict=True):
                connection.send(('result', result))
            print(json.dumps(dict(event='transformers_rollout_batch', batch_size=len(batch),
                                  max_prompt_tokens=max(len(item[1]['prompt_ids']) for item in batch))), flush=True)
        except Exception as error:
            for connection, _, _ in batch:connection.send(('error', f'{type(error).__name__}: {error}'))


def collect_group(policy, cfg, task_id, seed, directory, with_reference=False, rollout_indices=None):
    context = multiprocessing.get_context('spawn')
    directory = Path(directory)
    results = {}
    active = {}
    processes = []
    indices=list(range(cfg.group_size)) if rollout_indices is None else list(rollout_indices)
    if len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<cfg.group_size for i in indices):raise ValueError('Invalid rollout indices')
    if rollout_indices is not None and with_reference:raise ValueError('Subset collection is inference-only')
    next_index = 0
    last_progress = time.monotonic()

    def start(index):
        parent, child = context.Pipe()
        output = str(directory / f'episode-{index:02d}.json')
        process = context.Process(target=worker, args=(child, asdict(cfg), policy.version,
                                                       task_id, seed+index, output))
        process.start();child.close()
        processes.append(process);active[parent] = (index, process)

    try:
        while next_index < min(cfg.rollout_workers, len(indices)):
            start(indices[next_index]);next_index += 1
        while active:
            ready = wait(list(active), timeout=1)
            if not ready:
                if any(not process.is_alive() for _, process in active.values()):
                    raise RuntimeError('Tau2 worker exited without a completion message')
                if time.monotonic()-last_progress > 600:raise TimeoutError('Tau2 worker stalled')
                continue
            # Briefly gather simultaneous actor requests for a real GPU batch.
            time.sleep(.03)
            ready = wait(list(active), timeout=0)
            pending = []
            for connection in ready:
                message = connection.recv();last_progress = time.monotonic()
                if message[0] == 'generate':
                    pending.append((connection, *message[1:]))
                elif message[0] == 'done':
                    index, process = active.pop(connection)
                    results[index] = message[1];connection.close();process.join(timeout=1)
                    if next_index < len(indices):start(indices[next_index]);next_index += 1
                else:raise RuntimeError('Tau2 worker failed: '+str(message[1]))
            for offset in range(0,len(pending),cfg.rollout_batch_size):
                serve_requests(policy,pending[offset:offset+cfg.rollout_batch_size])
        episodes = [json.loads(Path(results[i]).read_text()) for i in indices]
        if with_reference:
            from tau2_grpo.rl_rollout import atomic_json
            scored_actions=0
            print(json.dumps(dict(event='rl_replay_scoring_start',actions=sum(len(e['actions']) for e in episodes))),flush=True)
            for index, episode in enumerate(episodes):
                for action in episode['actions']:
                    action['replay_old_logprobs'] = policy.score(action).detach().cpu().tolist()
                    action['reference_logprobs'] = policy.score(action, reference=True).detach().cpu().tolist()
                    scored_actions+=1
                    if scored_actions%10==0:print(json.dumps(dict(event='rl_replay_scoring_progress',completed_actions=scored_actions)),flush=True)
                atomic_json(results[index], episode)
        return episodes
    finally:
        for connection in active:connection.close()
        for process in processes:
            if process.is_alive():process.terminate()
            process.join(timeout=5)
