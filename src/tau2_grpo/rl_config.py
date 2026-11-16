"""Strict settings for retail, turn-aware on-policy GRPO."""
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class RetailRLConfig:
    environment: str = "tau2"
    assistance_level: str = "none"
    assisted_adaptive: bool = False
    assisted_prefix_max_actions: int = 12
    curriculum_file: str = ""
    tau2_eval_task_ids: str = ""
    external_timeout_seconds: int = 180
    eval_every_groups: int = 30
    short_scenario: str = "email_lookup"
    scenario_bank: str = ""
    bank_sampling: str = "balanced"
    bank_view: str = "opening"
    medium_scenario_bank: str = ""
    bank_validation_worlds: str = ""
    backend: str = "aligntune"
    model_id: str = 'outputs/hf-export/tiny-aya-global-function-calling-sft-525'
    initial_adapter: str = 'outputs/sft-retail-long-500-after-copy-repair/run-cbf90bb883/checkpoint-75'
    agent_prompt_file: str = ''
    user_prompt_file: str = ''
    system_prompt_file: str = ''
    chat_template_file: str = ''
    split_manifest: str = 'outputs/retail-data/split_manifest.json'
    output_dir: str = 'outputs/retail-rl'
    user_base_url: str = 'http://127.0.0.1:8004/v1'
    user_model: str = 'qwen3.5-4b'
    external_tokenizer: str = 'Qwen/Qwen3.5-4B'
    external_revision: str = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    external_context_limit: int = 16384
    user_max_tokens: int = 512
    summary_max_tokens: int = 2000
    context_limit: int = 8192
    response_tokens: int = 512
    prompt_budget: int = 7680
    max_actions: int = 40
    reward_mode: str = "native"
    argument_reward_weight: float = 0.5
    incorrect_argument_penalty: float = 0.25
    failed_call_penalty: float = 0.25
    failed_call_penalty_cap: float = 1.0
    max_episode_tokens: int = 16384
    group_size: int = 4
    rollout_workers: int = 1
    rollout_engine: str = 'transformers'
    rollout_base_url: str = 'http://127.0.0.1:8005/v1'
    rollout_batch_size: int = 1
    replay_batch_size: int = 1
    train_steps: int = 10
    updates_per_group: int = 1
    learning_rate: float = 5e-6
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    max_grad_norm: float = 1.0
    temperature: float = 0.8
    top_p: float = 1.0
    seed: int = 42
    precision: str = 'bf16'
    device: str = 'cuda'
    score_chunk_size: int = 16
    replay_tolerance: float = 0.02
    wandb_project: str = 'tau2-retail'
    wandb_mode: str = 'online'


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def load_settings(filename, overrides=None):
    raw = json.loads(Path(filename).read_text())
    raw.update(overrides or {})
    defaults = asdict(RetailRLConfig())
    for key, value in raw.items():
        if key not in defaults:
            raise ValueError('Unknown RL setting: '+key)
        expected = type(defaults[key])
        if expected is float:
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(key+' must be finite')
        elif type(value) is not expected:
            raise ValueError(key+' has incorrect type')
    cfg = RetailRLConfig(**raw)
    if cfg.assistance_level not in ('none','find','finish'):raise ValueError('Unknown assistance level')
    if cfg.bank_view not in ('opening','midturn_mix','medium_mix'):raise ValueError('Unknown bank_view')
    if (cfg.bank_view=='midturn_mix')!=(cfg.bank_sampling=='midturn_mix'):raise ValueError('Midturn view and sampler must be enabled together')
    if (cfg.bank_view=='medium_mix')!=(cfg.bank_sampling=='medium_mix'):raise ValueError('Medium view and sampler must be enabled together')
    if cfg.bank_view=='medium_mix' and not cfg.medium_scenario_bank:raise ValueError('Medium bank path is required')
    if cfg.bank_sampling not in ('balanced','clarification_mix','midturn_mix','medium_mix'):raise ValueError('Unknown bank_sampling')
    if cfg.bank_sampling!='balanced' and cfg.environment!='short_bank':raise ValueError('Curriculum sampling requires short_bank')
    if cfg.rollout_engine not in ('transformers', 'vllm'): raise ValueError('Unknown rollout_engine')
    if cfg.environment not in ('tau2','short_retail','short_bank'):raise ValueError('Unknown environment')
    if cfg.environment in ('short_retail','short_bank') and cfg.reward_mode!='short_verifiable':raise ValueError('Short tasks require short_verifiable rewards')
    if cfg.environment=='tau2' and cfg.reward_mode=='short_verifiable':raise ValueError('Short rewards cannot label native Tau2 episodes')
    if cfg.reward_mode not in ('native','additive_grounded','short_verifiable'): raise ValueError('Unknown reward_mode')
    if min(cfg.argument_reward_weight,cfg.incorrect_argument_penalty,cfg.failed_call_penalty,cfg.failed_call_penalty_cap)<0: raise ValueError('Reward weights must be nonnegative')
    if cfg.backend != 'aligntune': raise ValueError('Retail RL backend must be aligntune')
    for key in ('context_limit', 'response_tokens', 'max_actions', 'max_episode_tokens', 'group_size', 'rollout_workers', 'rollout_batch_size', 'replay_batch_size', 'train_steps', 'updates_per_group', 'external_context_limit', 'user_max_tokens', 'summary_max_tokens', 'score_chunk_size'):
        if getattr(cfg, key) < 1:
            raise ValueError(key+' must be positive')
    if not 0 < cfg.prompt_budget <= cfg.context_limit-cfg.response_tokens:
        raise ValueError('Prompt budget must preserve output reserve')
    if cfg.precision not in ('fp32', 'bf16') or cfg.device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported device/precision')
    if cfg.wandb_mode not in ('online', 'offline'):
        raise ValueError('W&B must be enabled')
    if cfg.learning_rate <= 0 or cfg.temperature < 0 or not 0 < cfg.top_p <= 1 or not 0 < cfg.clip_epsilon < 1:
        raise ValueError('Invalid optimization settings')
    if min(cfg.kl_beta, cfg.replay_tolerance) < 0 or cfg.max_grad_norm <= 0:
        raise ValueError('Invalid KL/gradient/replay settings')
    return cfg


def task_ids(cfg, partition='rl'):
    if cfg.environment=='short_bank':
        from tau2_grpo.rl_bank import bank_task_ids
        return bank_task_ids(cfg,partition)
    manifest = json.loads(path(cfg.split_manifest).read_text())
    if manifest['domain'] != 'retail':
        raise ValueError('Expected retail task manifest')
    pools = manifest['partitions']
    seen = set()
    for pool in pools.values():
        if seen.intersection(pool):
            raise ValueError('Task partitions overlap')
        seen.update(pool)
    return list(pools[partition])
