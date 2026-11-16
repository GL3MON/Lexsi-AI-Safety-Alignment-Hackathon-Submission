"""Strict JSON configuration for function-calling supervised fine-tuning."""

import json
import math
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SFTSettings:
    backend: str = "aligntune"
    model_id: str = "CohereLabs/tiny-aya-global"
    model_revision: str = "af89d219b53ed9b13b8a4645f9c8028973510324"
    initial_adapter: str | None = None
    chat_template_path: str | None = None
    dataset_path: str = "outputs/function-calling-data/sft.jsonl"
    eval_dataset_path: str | None = None
    task_split_manifest: str | None = None
    rl_holdout_path: str | None = "outputs/function-calling-data/rl_conversations.jsonl"
    output_dir: str = "outputs/sft"
    train_order: str = "random"
    preprocessing_num_workers: int = 1
    seed: int = 42
    validation_fraction: float = 0.1
    max_length: int = 6144
    rope_type: str | None = None
    rope_factor: float | None = None
    rope_original_max_position_embeddings: int | None = None
    rope_attention_factor: float | None = None
    overlength_policy: str = "error"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_train_epochs: float = 3.0
    max_steps: int = -1
    batch_size: int = 2
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    precision: str = "bf16"
    use_cpu: bool = False
    gradient_checkpointing: bool = True
    activation_offloading: bool = False
    attn_implementation: str = "sdpa"
    loss_type: str = "chunked_nll"
    example_normalized_loss: bool = False
    loss_chunk_size: int = 32
    mlp_chunk_size: int = 0
    loss_projection_dtype: str = "native"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    logging_steps: int = 1
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    evaluate_before_training: bool = False
    overfit_debug: bool = False
    resume_from_checkpoint: str | None = None
    wandb_project: str = "tiny-aya-function-calling"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["sft", "hermes", "function-calling"])
    log_sample_count: int = 2
    sample_max_new_tokens: int = 128


def _matches(value, annotation):
    origin = get_origin(annotation)
    if origin is types.UnionType:
        return any(_matches(value, option) for option in get_args(annotation))
    if annotation is type(None):
        return value is None
    if origin is list:
        return isinstance(value, list) and all(_matches(v, get_args(annotation)[0]) for v in value)
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is annotation


def load_settings(path, overrides=None):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("SFT config must be a JSON object")
    raw.update(overrides or {})
    hints = get_type_hints(SFTSettings)
    for key, value in raw.items():
        if key not in hints:
            raise ValueError(f"Unknown SFT setting: {key}")
        if not _matches(value, hints[key]):
            raise ValueError(f"Wrong type for SFT setting {key}: expected {hints[key]}")
    if raw.get("backend") == "agenttune":
        raw["backend"] = "aligntune"
    cfg = SFTSettings(**raw)
    if cfg.rope_type not in (None,'native','linear','dynamic','yarn'):
        raise ValueError('Unsupported rope_type')
    if cfg.rope_type is None and any(v is not None for v in (cfg.rope_factor,cfg.rope_original_max_position_embeddings,cfg.rope_attention_factor)):
        raise ValueError('RoPE parameters require an explicit rope_type')
    if cfg.rope_factor is not None and cfg.rope_factor<=1:raise ValueError('rope_factor must exceed one')
    if cfg.rope_original_max_position_embeddings is not None and cfg.rope_original_max_position_embeddings<1:raise ValueError('Original context must be positive')
    if cfg.rope_attention_factor is not None and (cfg.rope_attention_factor<=0 or cfg.rope_type!='yarn'):
        raise ValueError('Positive rope_attention_factor is supported only for YaRN')
    if cfg.task_split_manifest and (not cfg.eval_dataset_path or cfg.overfit_debug):
        raise ValueError("Task partitions require explicit evaluation data and overfit_debug=false")
    if cfg.train_order not in {"random", "tool_call_count"}:
        raise ValueError("train_order must be random or tool_call_count")
    if cfg.backend not in {"aligntune", "trl"}:
        raise ValueError("backend must be aligntune or trl")
    if not 0 < cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    for name in (
        "max_length", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
        "lora_rank", "lora_alpha", "logging_steps", "eval_steps", "save_steps",
        "save_total_limit", "sample_max_new_tokens",
        "loss_chunk_size",
        "preprocessing_num_workers",
    ):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_samples", "max_eval_samples"):
        if getattr(cfg, name) is not None and getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive or null")
    if cfg.mlp_chunk_size < 0:
        raise ValueError("mlp_chunk_size must be nonnegative")
    if cfg.max_steps != -1 and cfg.max_steps < 1:
        raise ValueError("max_steps must be -1 or positive")
    if cfg.num_train_epochs <= 0 or cfg.learning_rate <= 0 or cfg.max_grad_norm <= 0:
        raise ValueError("epochs, learning_rate and max_grad_norm must be positive")
    if cfg.weight_decay < 0 or not 0 <= cfg.warmup_ratio < 1 or not 0 <= cfg.lora_dropout < 1:
        raise ValueError("Invalid weight decay, warmup ratio or LoRA dropout")
    if cfg.log_sample_count < 0 or not cfg.lora_target_modules:
        raise ValueError("Need nonnegative log_sample_count and nonempty LoRA targets")
    if cfg.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16 or fp32")
    if cfg.use_cpu and cfg.precision != "fp32":
        raise ValueError("CPU training requires precision=fp32")
    if cfg.wandb_mode not in {"online", "offline"} or not cfg.wandb_project.strip():
        raise ValueError("W&B logging is required: set project and online or offline mode")
    if cfg.overlength_policy not in {"error", "drop"}:
        raise ValueError("overlength_policy must be error or drop")
    if cfg.loss_type not in {"chunked_nll", "nll"}:
        raise ValueError("loss_type must be chunked_nll or nll")
    if cfg.loss_projection_dtype not in {"native", "fp32"}:
        raise ValueError("loss_projection_dtype must be native or fp32")
    if cfg.load_best_model_at_end and cfg.save_steps % cfg.eval_steps:
        raise ValueError("save_steps must be a multiple of eval_steps when loading best model")
    return cfg


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def settings_dict(cfg):
    return asdict(cfg)
