"""Validated experiment settings; importing this module does not load models."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    # Qwen3 supplies a native tool-calling chat template. The 0.6B alternative
    # is cheaper for a wiring run but may struggle with real troubleshooting.
    model_id: str = "Qwen/Qwen3-1.7B"
    model_revision: str = "main"  # Resolved Hugging Face commit is recorded below.
    domain: str = "telecom"
    seed: int = 42
    group_size: int = 4
    train_steps: int = 10
    learning_rate: float = 5e-6
    lora_rank: int = 8
    # These are small experiment samples, not the whole benchmark. Increase
    # them for a serious evaluation after checking throughput and VRAM.
    fit_scan_tasks: int = 8
    validation_tasks: int = 4
    final_test_tasks: int = 4
    validation_fraction: float = 0.2
    max_actions: int = 16
    max_tool_iterations: int = 16
    # TRL 1.7.1 counts both generated text AND inserted tool results toward
    # this completion budget. A small budget can prevent completing a ticket.
    max_completion_tokens: int = 6144
    context_limit: int = 32768
    # Keep full tool results by default. The trainer's token budget ends overly
    # long rollouts; we do not silently cut the task ticket or domain policy.
    temperature: float = 0.8
    top_p: float = 0.95
    kl_beta: float = 0.02
    validation_regression_tolerance: float = 0.0


def load_config(path: Path) -> ExperimentConfig:
    """Load strict JSON settings and reject invalid training budgets."""
    raw = json.loads(path.read_text())
    defaults = ExperimentConfig()
    for key, value in raw.items():
        if not hasattr(defaults, key):
            raise ValueError(f"Unknown configuration key: {key}")
        expected = type(getattr(defaults, key))
        if expected is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        elif type(value) is not expected:
            raise ValueError(f"{key} must be {expected.__name__}")
    cfg = ExperimentConfig(**raw)
    if cfg.domain not in {"telecom", "retail"}:
        raise ValueError("Supported domains are telecom and retail (preview only)")
    if cfg.group_size < 2 or not 0 < cfg.validation_fraction < 1:
        raise ValueError("group_size must be >= 2 and validation_fraction in (0, 1)")
    for key in (
        "train_steps",
        "max_actions",
        "max_tool_iterations",
        "fit_scan_tasks",
        "validation_tasks",
        "final_test_tasks",
        "max_completion_tokens",
        "context_limit",
        "lora_rank",
    ):
        if getattr(cfg, key) < 1:
            raise ValueError(f"{key} must be positive")
    if cfg.max_tool_iterations < cfg.max_actions:
        raise ValueError("max_tool_iterations must cover max_actions")
    if cfg.max_completion_tokens >= cfg.context_limit:
        raise ValueError("context_limit must leave space for the prompt")
    if cfg.learning_rate <= 0 or cfg.temperature <= 0 or not 0 < cfg.top_p <= 1:
        raise ValueError("Invalid learning rate or sampling settings")
    if cfg.kl_beta < 0 or cfg.validation_regression_tolerance < 0:
        raise ValueError("KL and regression tolerance must be nonnegative")
    return cfg
