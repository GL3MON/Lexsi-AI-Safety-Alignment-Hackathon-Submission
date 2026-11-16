"""Isolated Tau2 workers sharing one colocated vLLM LoRA engine.

Only adapters cross the trainer/server boundary. Each rollout group completes
before any weight update or adapter replacement, so behavior versions cannot mix.
"""
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from urllib.request import Request, urlopen


def adapter_name(version):
    return f'policy_v{version}'


def adapter_request(base_url, operation, payload):
    request = Request(base_url.rstrip('/') + '/' + operation,
                      data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=180) as response:
        return response.read().decode()


def sync_adapter(policy, cfg, run):
    """Called only after all workers have joined; use a unique name per version."""
    destination = Path(run) / 'rollout-adapters' / adapter_name(policy.version)
    policy.model.save_pretrained(destination, selected_adapters=['policy'])
    adapter = destination / 'policy'
    if not (adapter / 'adapter_config.json').is_file():
        raise RuntimeError('Exported policy adapter is missing')
    # Free the sole vLLM adapter slot before loading the replacement.
    if policy.version:
        adapter_request(cfg.rollout_base_url, 'unload_lora_adapter',
                        {'lora_name': adapter_name(policy.version - 1)})
    adapter_request(cfg.rollout_base_url, 'load_lora_adapter',
                    {'lora_name': adapter_name(policy.version), 'lora_path': str(adapter.resolve())})


class VLLMPolicy:
    """HTTP-only sampler: deliberately has no score() or CUDA model."""
    def __init__(self, cfg, version, tokenizer, client):
        self.cfg, self.version, self.tokenizer, self.client = cfg, version, tokenizer, client

    def generate(self, prepared, max_tokens, seed):
        eos = self.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError('An explicit trained EOS token is required')
        choice = self.client.completions.create(
            model=adapter_name(self.version), prompt=prepared['prompt_ids'],
            max_tokens=max_tokens, temperature=self.cfg.temperature,
            top_p=self.cfg.top_p, seed=seed, logprobs=1,
            extra_body={'return_tokens_as_token_ids': True, 'ignore_eos': True,
                        'stop_token_ids': [eos], 'skip_special_tokens': False,
                        'include_stop_str_in_output': True, 'top_k': -1,
                        'repetition_penalty': 1.0}).choices[0]
        if choice.logprobs is None:
            raise RuntimeError('vLLM omitted sampled-token log probabilities')
        pieces, logs = choice.logprobs.tokens, choice.logprobs.token_logprobs
        if not pieces or len(pieces) != len(logs):
            raise RuntimeError('vLLM returned unaligned token probabilities')
        ids = []
        for piece, value in zip(pieces, logs, strict=True):
            if not piece.startswith('token_id:') or value is None or not math.isfinite(value):
                raise RuntimeError('Expected exact vLLM token IDs and finite log probabilities')
            ids.append(int(piece.removeprefix('token_id:')))
        return dict(text=self.tokenizer.decode(ids, skip_special_tokens=True),
                    completion_ids=ids, old_logprobs=logs,
                    ended_with_eos=ids[-1] == eos,
                    decoding_mode='sampled' if self.cfg.temperature else 'greedy',
                    top_p=self.cfg.top_p, sampling_seed=seed,
                    generation_engine='vllm', rollout_adapter=adapter_name(self.version))


def _worker(payload):
    # Spawn rather than fork: Tau2 patches global registries and evaluators.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from openai import OpenAI
    from transformers import AutoTokenizer
    from tau2_grpo.rl import make_episode
    from tau2_grpo.rl_config import RetailRLConfig, path
    raw_cfg, version, task, seed, output = payload
    cfg = RetailRLConfig(**raw_cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(path(cfg.initial_adapter)))
    if cfg.chat_template_file:
        tokenizer.chat_template = path(cfg.chat_template_file).read_text()
    with OpenAI(base_url=cfg.rollout_base_url, api_key='local', timeout=300, max_retries=0) as client:
        policy = VLLMPolicy(cfg, version, tokenizer, client)
        make_episode(policy, cfg, task, seed, output, with_reference=False)
    return output


def collect_group(policy, cfg, task, seed, directory, with_reference=True):
    directory = Path(directory)
    payloads = [(asdict(cfg), policy.version, task, seed+i,
                 str(directory / f'episode-{i:02d}.json')) for i in range(cfg.group_size)]
    with ProcessPoolExecutor(max_workers=min(cfg.rollout_workers, cfg.group_size),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        outputs = list(pool.map(_worker, payloads))
    episodes = [json.loads(Path(output).read_text()) for output in outputs]
    # Teacher-forced old/reference probabilities use the trainer's exact contexts.
    # vLLM's actual behavior probabilities remain separate for IS correction.
    from tau2_grpo.rl_rollout import atomic_json
    for output, episode in zip(outputs, episodes, strict=True):
        if not with_reference:
            continue
        for action in episode['actions']:
            action['replay_old_logprobs'] = policy.score(action).detach().cpu().tolist()
            action['reference_logprobs'] = policy.score(action, reference=True).detach().cpu().tolist()
        atomic_json(output, episode)
    return episodes
