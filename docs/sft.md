# Function-calling SFT

This entry point trains a LoRA adapter for `CohereLabs/tiny-aya-global` using the
prepared Hermes conversations. Its default AgentTune backend orchestrates native
EventLog-based SFT through `Project.train`; `--backend trl` selects direct TRL.
Both use the same installed TRL `SFTTrainer` optimizer runner and log through
an explicit `wandb.init` and the Trainer's `WandbCallback`. See the
[backend design and MIG 2 smoke commands](sft-backends.md).
Three-sample Tiny Aya GPU smoke and overfit diagnostics are supported. Full-dataset
training is separate from those diagnostics.

## Start

Run from the project root; configured relative paths resolve against that root.

```bash
# Review all resolved settings without loading libraries/model weights.
.venv/bin/python -m tau2_grpo.sft --config configs/sft.json --print-config

# Check the entire dataset and all assistant-only masks without loading weights.
make sft-prepare

# Authenticate once; the trainer also supports the normal WANDB_API_KEY variable.
.venv/bin/wandb login

# Train and log to the W&B cloud. CUDA_VISIBLE_DEVICES may be a MIG UUID.
CUDA_VISIBLE_DEVICES=0 make sft
```

The executable launcher also works from any directory:

```bash
./scripts/run_sft.sh
./scripts/run_sft.sh --prepare-only
./scripts/run_sft.sh --config /absolute/path/to/custom.json
./scripts/run_sft.sh --set learning_rate=0.0001 --set num_train_epochs=2
```

It uses this project's `.venv/bin/python`, preserves `CUDA_VISIBLE_DEVICES` and
W&B credentials, and forwards all CLI arguments. `SFT_CONFIG` optionally changes
the default config path; an explicit `--config` takes precedence.

The trainer prints its artifact directory and W&B run URL. Default project:
`tiny-aya-function-calling`; optionally set `wandb_entity` to your user/team.
W&B starts before allocating model weights. Authentication/init failures stop the
run; there is no silent fallback to disabled logging. Do not put API keys in JSON
configs. `WANDB_DISABLED=true` is rejected. This standalone entry point explicitly
sets the configured W&B mode, so it does not inherit GRPO's disabled logging mode.

For a short real-model GPU check that still logs online:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tau2_grpo.sft \
  --config configs/sft.json --max-steps 2 \
  --set max_train_samples=8 --set max_eval_samples=2 \
  --set gradient_accumulation_steps=1 --set warmup_ratio=0 \
  --set eval_steps=1 --set save_steps=1
```

This check uses the real base model and takes GPU memory/time. It is not a
function-calling quality evaluation.

## Configure

All defaults are explicit in `configs/sft.json`. Unknown fields, wrong types,
disabled W&B logging, and invalid budgets fail validation. Override any setting
without editing the file using `--set KEY=JSON`. Strings need JSON quotes:

```bash
.venv/bin/python -m tau2_grpo.sft --config configs/sft.json \
  --set learning_rate=0.0001 --set num_train_epochs=2 \
  --set wandb_run_name='"aya-hermes-lora-v1"'
```

| Settings | Purpose / defaults |
| --- | --- |
| `backend` | `agenttune` default; `trl` selects direct TRL. Also available as `--backend`. |
| `model_id`, `model_revision` | Tiny Aya at a pinned revision. Change both when switching models. |
| `dataset_path`, `rl_holdout_path` | SFT input and separately reserved RL pool. The latter is checked for group overlap, never trained on. Set null only when intentionally using a different dataset without an RL pool. |
| `output_dir` | Parent of a fresh `run-<id>` directory. |
| `validation_fraction`, `seed` | 10% group-based validation inside the SFT pool, seed 42. Related single/multi-turn conversations stay together. |
| `max_train_samples`, `max_eval_samples` | Optional deterministic caps after splitting, useful for quick checks. |
| `max_length`, `overlength_policy` | 6,144 tokens; error on overlong examples. `drop` explicitly excludes and records them. No silent truncation. |
| `num_train_epochs`, `max_steps` | Three epochs. Positive `max_steps` overrides epochs; -1 uses epochs. |
| `batch_size`, `gradient_accumulation_steps` | 2 × 4 = effective batch 8 on one GPU, as requested. Batch 2 failed the longest-example probe under the measured shared-GPU memory budget; use batch 1 if that budget is unchanged. |
| `learning_rate`, `lr_scheduler_type`, `warmup_ratio` | 2e-4, cosine, 3%; ratio is converted to warmup optimizer steps for the installed Transformers API. |
| `weight_decay`, `max_grad_norm`, `optim` | 0.01, 1.0, `adamw_torch`. |
| `precision`, `use_cpu` | bf16 CUDA by default. CPU checks require fp32. Exactly one process/GPU is supported. |
| `gradient_checkpointing`, `attn_implementation` | Non-reentrant gradient checkpointing and SDPA. |
| `activation_offloading` | `configs/sft.json` enables activation offloading to CPU so the longest retained examples fit this shared MIG slice. Disable only after checking memory for your sequence lengths. |
| `loss_type` | `chunked_nll`, reducing peak memory by projecting only supervised positions in chunks. `nll` is also supported. |
| `loss_chunk_size`, `loss_projection_dtype` | 32 supervised tokens per loss chunk; native-dtype head projection with FP32 cross entropy. Avoids TRL's temporary FP32 copy of Tiny Aya's large vocabulary head. `fp32` restores the original projection implementation. This integration checks for TRL 1.7.1. |
| `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules` | Rank 16, alpha 32, dropout 0.05, attention and MLP projections. Base weights remain frozen. |
| `logging_steps`, `eval_steps`, `save_steps` | Log each optimizer step; evaluate and checkpoint every 25. |
| `save_total_limit`, `load_best_model_at_end` | Keep two checkpoints and select lowest validation loss. Save interval must be a multiple of evaluation interval. |
| `evaluate_before_training` | Optional baseline loss before training; false by default. |
| `overfit_debug` | False by default. When explicitly true, validation intentionally equals the capped training subset to diagnose memorization. These metrics do not measure generalization. |
| `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags` | W&B destination and run labeling. |
| `wandb_mode` | `online` default; explicit `offline` still writes real W&B event files. Disabled mode is unsupported. |
| `log_sample_count`, `sample_max_new_tokens` | Log two fixed validation generations after training, capped at 128 new tokens. Set count to 0 to omit sample generation. |
| `resume_from_checkpoint` | Optional Trainer checkpoint path; also available as `--resume`. |

## Labels and data checks

Tiny Aya has no assistant-generation masks in its supplied chat template. We
render the original template and derive each assistant span using exact token
prefixes before and after that response. Every prefix must match the full token
sequence; incompatible templates fail clearly. User text, system text, tool
observations, and assistant role headers receive label -100. Assistant responses
and their turn-ending tokens receive real labels. The Trainer consumes these
prebuilt labels without rebuilding or truncating them. Its native
`assistant_only_loss` switch is therefore false; assistant-only loss is implemented
in the explicit labels, not that switch.

The prepared SFT dataset has 1,397 training and 155 validation conversations with
default settings. All mask boundaries passed using the actual pinned Tiny Aya
tokenizer. The reserved 388-conversation RL pool is unchanged. Changing the data,
seed or sample caps changes these counts and is recorded in the run artifacts.

## W&B logging and artifacts

The dashboard receives training/evaluation loss, token accuracy, entropy, gradient
norm, learning rate, runtime metrics, full non-secret configuration, data counts,
and trainable parameter counts. Tables show the label-mask examples and final
validation generations beside their references. Checkpoints/adapters are saved
locally; uploading model weights to W&B is not enabled by this entry point.

Each run saves:

- `config.json`, `runtime.json`: resolved settings, package/model revision, W&B ID/URL.
- `data_summary.json`, `data_manifest.json`: hashes, split membership, token counts,
  excluded examples and masking audit.
- `mask_preview.json`: decoded assistant targets and their exact token spans.
- `checkpoint-<step>/`: adapter and Trainer state, optimizer, scheduler and RNG state.
- `final_adapter/`: selected trained adapter and tokenizer, not a merged base model.
- `metrics.json`, `trainer_log_history.json`, `generated_samples.json`.
- `wandb/`: W&B event files and diagnostic logs, including for offline runs.

Training completion requires finite training loss and an observed change in LoRA
weights. Validation loss is a training diagnostic; it does not establish success
on executed retail tool tasks.

To log offline explicitly and sync later:

```bash
.venv/bin/python -m tau2_grpo.sft --config configs/sft.json --wandb-mode offline
.venv/bin/wandb sync outputs/sft/run-<id>/wandb/offline-run-<timestamp>-<wandb-id>
```

## Resume

```bash
.venv/bin/python -m tau2_grpo.sft --config configs/sft.json \
  --resume outputs/sft/run-<id>/checkpoint-50
```

Resume uses the checkpoint's parent run directory. Tokenized train/eval data must
match exactly; model, LoRA and W&B destination settings must also match. Online
mode resumes the saved W&B run ID with `resume=must`. Offline W&B writes another
local event directory. Resume configuration is saved separately. When continuing
beyond the previous budget, set a higher `--max-steps`.

## Implementation checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_sft.py
```

For measured batch-size limits on the current GPU, run:

```bash
.venv/bin/python scripts/probe_sft_batch_size.py --config configs/sft.json
```

The probe runs two real optimizer steps for each candidate batch in a fresh child
process, using repetitions of the longest selected conversation. It uses the same
LoRA, gradient-checkpointing and loss settings and logs results to W&B. It tests
powers of two, then binary-searches the failed/successful interval. Output includes
peak allocated/reserved memory, free memory before loading the model, and failures.
`--sample-manifest <run>/data_manifest.json` restricts the workload to a run's
training samples. A result depends on sequence length and current competing GPU
usage; it is a measured boundary, not a universal batch-size guarantee. The probe
does not modify the default training batch size. `--max-batch-size` caps the search;
if the cap succeeds, the result is a lower bound rather than an exact maximum.

The integration script constructs a tiny random local Llama model and synthetic
fixture data, runs the real trainer on CPU with W&B offline, and checks optimizer
updates, evaluation/checkpoints, the saved adapter, and W&B event files. It never
trains the Tiny Aya base model or verifies cloud authentication. The separately
executed Tiny Aya smoke, overfit and batch probes are described in
[GPU diagnostic results](sft-diagnostics.md). Full-dataset quality still requires
a full training run and held-out evaluation.
