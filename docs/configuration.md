# Configuration reference

`ExperimentConfig` in `src/tau2_grpo/config.py` defines immutable defaults.
Each preset is a JSON object containing overrides; unspecified fields retain
defaults. Unknown keys and mismatched primitive types are rejected. Integer fields
reject booleans; float fields accept finite JSON numbers. File/JSON parse errors
surface directly through the CLI.

```bash
.venv/bin/tau2-grpo config --config configs/small.json
cp configs/small.json configs/my-run.json
# Edit my-run.json, then inspect the merged settings before training.
.venv/bin/tau2-grpo config --config configs/my-run.json
```

## Every setting

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `model_id` | `Qwen/Qwen3-1.7B` | Hugging Face model identifier; string. Tool-calling template and LoRA target compatibility are required at runtime. |
| `model_revision` | `main` | Requested model revision; resolved revision is reused for baseline/training/reload and saved when available. |
| `domain` | `telecom` | Only accepted domain in this integration. |
| `seed` | `42` | Integer for split shuffling, Gym resets and model sampling/training seeds. |
| `group_size` | `4` | Generations per task, at least 2; also controls accumulation, generation scheduling and evaluation batch size. |
| `train_steps` | `10` | Positive optimizer-step limit, not episode or epoch count. |
| `learning_rate` | `0.000005` | Positive learning rate for adapter training. |
| `lora_rank` | `8` | Positive LoRA rank; alpha is derived as twice this value. |
| `fit_scan_tasks` | `8` | Positive maximum number of fitting-partition tasks scanned for baseline failures. |
| `validation_tasks` | `4` | Positive maximum number of reserved validation tasks used by the gate. |
| `final_test_tasks` | `4` | Positive maximum number of official test tasks used after the gate. |
| `validation_fraction` | `0.2` | Fraction of official train reserved for validation; strictly between 0 and 1. |
| `max_actions` | `16` | Positive wrapper action budget; malformed argument JSON consumes an action too. |
| `max_tool_iterations` | `16` | Positive TRL tool-calling iteration limit; must be at least `max_actions`. |
| `max_completion_tokens` | `6144` | Positive rollout completion budget; inserted tool results count under the pinned TRL behavior. Must be below `context_limit`. |
| `context_limit` | `32768` | Positive total context budget; runtime checks model support and prompt plus completion reserve. |
| `temperature` | `0.8` | Positive sampling temperature. |
| `top_p` | `0.95` | Nucleus sampling probability in (0, 1]. |
| `kl_beta` | `0.02` | Nonnegative KL penalty coefficient passed as TRL `beta`. |
| `validation_regression_tolerance` | `0.0` | Nonnegative tolerance passed to AgentTune's gate. Local strict-improvement and changed-weight guards still apply. |

Task sample sizes use Python slicing. Asking for more tasks than available uses
all available tasks; it does not synthesize data. Validation count is
`max(1, round(train_count * validation_fraction))`; a tiny fitting partition can
become empty and fail the workflow's nonempty assertions. Counts are not a
replacement for checking `split_manifest.json`.

## Presets

| Setting | `local.json` | `small.json` |
| --- | --- | --- |
| Model | Qwen3-1.7B | Qwen3-0.6B |
| Group size | 4 | 2 |
| Optimizer steps | 10 | 1 |
| Fitting scan | 8 tasks | 2 tasks |
| Gate sample | 4 tasks | 1 task |
| Final test sample | 4 tasks | 1 task |

All other settings are identical defaults. The small preset retains the full
context and rollout budgets, so it is not a memory guarantee or a useful estimate
of benchmark-wide quality.

## Derived and fixed training settings

The workflow sets `backend="trl"`, `use_vllm=False`, and native
`environment_factory=Tau2SoloRollout`. No Unsloth backend is selected.

- Per-device training batch is 1; accumulation steps and steps per generation
  equal `group_size`. Both training and evaluation generations equal group size.
- Evaluation batch size equals group size. Tasks can have uneven episode counts
  due to padding; final summaries average within each task before averaging tasks.
- LoRA targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`; alpha is `2 * rank`,
  dropout is 0, bias is disabled and task type is causal language modeling.
- Training uses SDPA, gradient checkpointing with `use_reentrant=False`, BF16
  when supported and FP16 otherwise. Model initialization disables cache;
  generation enables it.
- Qwen thinking is disabled in chat-template kwargs. Changing the model may
  require corresponding template and target-module changes.
- Loss type is `grpo`, reward scaling is `group`, unused columns are retained,
  and truncated-completion losses are not automatically discarded.
- Warmup steps are 0; logs are emitted every step; checkpoints are saved every
  `max(1, train_steps // 2)` steps with a limit of two checkpoints.
- Automatic training-time evaluation is disabled. Explicit baseline, validation
  and final-test phases call `trainer.evaluate()`.
- Data-loader workers are 0. Reporting and Hub pushes are disabled.
- Gym uses solo mode, substring communication scoring and
  `max_steps=2 * max_actions + 4`. No JSON option changes that scoring protocol.

These fixed settings require editing `experiment.py` or `rollout.py`, rather
than adding undocumented JSON fields.

## CLI paths and environment variables

`--config` defaults to the project's `configs/local.json`, `--tau2` to the sibling
`tau2-bench`, and `--output` to the project's `outputs`. Explicit relative paths
resolve from the launching working directory. The source-based project discovery
assumes an editable/source installation; a relocated wheel is not supported.

| Variable | Behavior |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Defaults to `0` if absent; supplied selection is retained. Set before process startup. |
| `WORLD_SIZE` | Training requires 1; defaults to 1 when absent. |
| `TOKENIZERS_PARALLELISM` | Defaults to `false`. |
| `WANDB_MODE` | Defaults to `disabled`; trainer also uses `report_to="none"`. |
| `PURE_TRL_MODE` | Defaults to `1`. |
| `TAU2_CHECKOUT` | Setup override for checkout location; CLI sets its own internal value from `--tau2`. |
| `TAU2_GRPO_CONFIG` | Internal resolved config path set by the CLI. |
| `TAU2_GRPO_OUTPUT` | Internal resolved output root set by the CLI. |
| `TAU2_GRPO_SMOKE` | Internal CLI flag: `1` exits after the Gym smoke check. |
| `UV_FROZEN` | Set to 1 by the installation script. |

Hosted LLM API keys are not required by the intended solo/substring workflow.
Model downloads use the normal Hugging Face environment/cache behavior; private
or gated replacements may need their own download credentials. The project does
not load a project `.env` itself. Optional behavior inherited from the installed
tau2 configuration can introduce additional dependencies or services.
