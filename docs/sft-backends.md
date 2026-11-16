# AgentTune and direct TRL SFT backends

`configs/sft.json` now selects `"backend": "agenttune"`. The existing command and
all training settings still work:

```bash
./scripts/run_sft.sh --set batch_size=1 --set gradient_accumulation_steps=8
```

Select the backend explicitly with either command:

```bash
./scripts/run_sft.sh --backend agenttune
./scripts/run_sft.sh --backend trl
```

The same option is available as `--set 'backend="trl"'` or the JSON field. The
convenience `--backend` flag takes precedence over JSON/`--set`. No automatic
fallback changes the requested backend. GPU selection remains controlled by
`CUDA_VISIBLE_DEVICES`; changing backends does not change batch size, accumulation,
LoRA, optimizer, dataset, W&B destination, or loss settings.

## Actual execution paths

The implementation follows AgentTune's documented
[SFT trainer-factory contract](https://github.com/Lexsi-Labs/AgentTune/blob/d4db4c12d52342227c4113a70183d0f64aa2f345/docs/user-guide/distillation.md#5-distill-vs-trainfmtsft)
and the installed public `Project.train(fmt="sft")` implementation. AgentTune
1.1.3 is pinned to commit `d4db4c12d52342227c4113a70183d0f64aa2f345`.

- AgentTune: authored reference conversations → native `EventLog` objects →
  `Project.train(fmt="sft")` → AgentTune's projected SFT rows → a trainer factory
  → TRL `SFTTrainer` → optimizer steps. AgentTune owns the call to `.train()` and
  emits its native `train/started` and `train/done` lifecycle events.
- Direct TRL: the same verified tokenized conversations → `SFTTrainer` → optimizer
  steps. This path does not import or require AgentTune.

Using a custom trainer factory is the documented AgentTune integration, not a
replacement for its public lifecycle. The shared factory preserves our validated
assistant-only masks and memory-efficient loss. It does not use the legacy
`agenttune.core.sft` path or fabricate an unsupported `create_agentic_trainer("sft")`.

## Preserving the dataset and objective

AgentTune's EventLog does not have a system-message event, so the adapter records
the original role and restores it after native projection. Assistant targets are
stored as `TEXT` events with the exact original text: using `TOOL_CALL` projection
would reserialize JSON and split parallel calls, changing the training targets.
Tagged tool observations use `TOOL_RESULT` events and are mapped back to Tiny Aya's
user-role observations. All assistant segments retain weight 1, matching direct TRL.

The factory consumes the rows actually provided by AgentTune, verifies every
message and message-level mask, retokenizes them, and requires exact equality with
the original token IDs and labels before creating the optimizer runner. It never
ignores projected rows and silently trains on a different dataset. These are
hand-authored SFT demonstrations; no policy logprobs or on-policy rollouts are
invented. A failed projection check stops training.

## Output compatibility and W&B

Both backends retain `config.json`, `runtime.json`, `data_manifest.json`,
`data_summary.json`, `mask_preview.json`, `checkpoint-*`, `final_adapter/`,
`metrics.json`, `trainer_log_history.json`, and `generated_samples.json`.
Online W&B logging remains mandatory by default, including sample tables.

Additional audit outputs:

- `backend_audit.json`: selected backend, data fingerprint, and AgentTune projection
  equality/count checks when applicable.
- `agenttune_events.json`: actual AgentTune lifecycle events, including training
  row/trajectory counts and the optimizer result. Only the AgentTune path writes it.
- `runtime.json`: backend, AgentTune version/revision when selected, and the exact
  `CUDA_VISIBLE_DEVICES` used by the process.

Checkpoints still use the same Trainer/PEFT format and existing resume command.
Resuming requires identical tokenized training/validation data and matching model
and LoRA settings. Older direct-TRL checkpoints can be resumed through either
backend because the optimizer runner and data contract are shared.

Output formats and supervision are identical. The CPU parity integration compares
adapter tensor values, training loss, and generated samples exactly. GPU timings
and low-precision numerical results need not be bit-identical, especially with
the optional 8-bit optimizer.

## MIG 2 smoke

```bash
./scripts/smoke_sft.sh --backend agenttune
./scripts/smoke_sft.sh --backend trl
```

This script resolves MIG device 2 to its UUID using `nvidia-smi -L`, overrides any
ambient `CUDA_VISIBLE_DEVICES`, and runs three real Tiny Aya training samples for
three steps. Set `SFT_SMOKE_GPU=<uuid>` only to select a different device explicitly.
W&B stays online. It uses batch 1, accumulation 1, activation offloading, and
`adamw_bnb_8bit` (bitsandbytes 0.50.2) because the current MIG 2 slice is occupied by
other workloads. This is a smoke configuration; the default training optimizer
remains `adamw_torch`. Command-line overrides are still forwarded.

The loss wrapper now preserves the base model forward signature so generation can
validate/use `logits_to_keep=1`. It avoids allocating logits for every prompt token
with Tiny Aya's large vocabulary, without changing next-token sampling semantics.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_sft.py --backend both
```

The CPU integration uses one local random tiny model and the same examples for
both backends, exercises real optimization/checkpointing/evaluation/W&B offline
events, and compares final adapter tensors, loss and generated samples. Unit tests
also cover native AgentTune lifecycle emission, multi-turn/system preservation,
resume argument forwarding, and rejection of projection drift.

Verified on 2026-11-09:

- 21 unit tests passed.
- CPU integration: byte-identical adapter tensors, equal training loss, and equal
  generated samples across backends; saved proof:
  `outputs/sft-validation-47f82098/parity.json`.
- Real Tiny Aya MIG 2 smoke: both backends completed all three steps, evaluation,
  checkpoint/adapter save, generated sample and online W&B sync. Input/label
  fingerprints, metric field names and generated samples matched. The 8-bit GPU
  optimizer produced slightly different loss values, as expected for this
  low-precision test. Each changed 504 adapter tensors and peaked at 7.016 GiB
  allocated / 7.100 GiB reserved.
- Comparison artifact: `outputs/sft-backend-smoke/comparison.json`.
- Online [AgentTune smoke](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/henlcvs7)
  and [direct TRL smoke](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/8jkycazj).
