# AlignTune migration

The active SFT, retail RL, and basic telecom GRPO paths use AlignTune pinned at `c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6`. AgentTune is removed from runtime dependencies and imports. Historical run artifacts retain their original backend metadata.

## Integration boundary

`tau2_grpo.aligntune_runtime` implements explicit `UnifiedTrainerBase` extensions with native AlignTune callbacks, training state, and provenance. The exact package origin/checkout commit is verified before training.

- SFT uses the `masked_sft` extension and TRL's optimizer loop. It preserves precomputed assistant labels, selected `loss_message_indices`, chunked NLL, continuation from adapters, resume forwarding, W&B, and sequential tool-call curriculum ordering. AlignTune's stock generation SFT currently selects message/prompt columns and discards precomputed label columns, so it is deliberately not used for these records.
- Retail RL uses the `turn_aware_tau2_grpo` extension. Actual Tau2 trajectories, frozen customer and summarizer, per-action contexts, sampled probabilities, numerical replay corrections, reference adapter, episode-balanced loss, and terminal rewards remain in the shared harness. AlignTune owns the training lifecycle and provenance; our specialized PyTorch/PEFT GRPO objective is retained. This is not stock flat-completion AlignTune GRPO.
- The basic telecom solo bridge uses the `telecom_native_environment_grpo` extension with TRL's native `environment_factory` support. The old AgentTune FullClosedLoop retraining/deployment service is retired; basic training and separate validation/test evaluation are retained. Preview and CPU bridge smoke commands remain available.

## Commands

Existing entry points are unchanged:

```bash
scripts/run_sft.sh --config configs/sft-retail.json --backend aligntune
scripts/run_sft.sh --config configs/sft-retail.json --backend trl
scripts/run_retail_rl.sh check
scripts/run_retail_rl.sh train --config configs/rl-retail.json
```

`backend: "aligntune"` is the SFT default. The historical SFT value/flag `agenttune` is accepted as an alias for `aligntune`; it never imports or requires AgentTune. `trl` remains a direct SFT diagnostic option. Retail RL requires the AlignTune backend. Its default initial adapter is the selected long-SFT checkpoint 75; override `initial_adapter` for another checkpoint.

Training keeps W&B online by default. Nucleus sampling is available for evaluation; GRPO training requires `top_p=1` and positive temperature so replay matches its sampling distribution. All-tied reward groups skip optimization; infrastructure failures retain null rewards.

## Artifacts

Existing SFT/rollout files remain available. New framework artifacts are:

- `aligntune_runtime.json`: pinned package revision, extension name, training outcome.
- `aligntune_events.json`: native lifecycle and metric callbacks.
- `lexsi_provenance.json`: AlignTune provenance.
- `backend_audit.json`: SFT backend and label fingerprint.

Original datasets and production checkpoints are unchanged. No full production RL run was launched for this migration.

## Verification

- 63 unit tests passed, including summary causality, masked context, behavior/replay alignment guards, and tied-group handling.
- `scripts/validate_sft.py --backend both`: tiny CPU-model parity passed with W&B online. AlignTune and direct TRL produced byte-identical adapters, identical training loss, and identical generated samples. Report: `outputs/sft-validation-9b95130f/parity.json`.
- Real Tiny Aya GPU SFT smoke: three training samples, one update, 504 changed adapter tensors. `outputs/aligntune-smoke/sft/run-f7336bf3b5/metrics.json`. The initial one-step smoke used a warmup-only zero-LR step and was correctly rejected; retry used zero warmup.
- Real Tiny Aya GPU GRPO optimizer smoke: one non-tied synthetic contrast update, 504 changed adapter tensors, finite gradients, zero policy replay mismatch. Explicitly synthetic rewards, not Tau2 task rewards. `outputs/aligntune-smoke/rl-optimizer/report.json`.
- Live Tau2 retail RL smoke: one group of two episodes on reserved RL task 59, one actor action each, real simulator/evaluator. Both terminal rewards were zero, and the update was correctly skipped. `outputs/aligntune-smoke/tau2-grpo/result.json`.
- Telecom CPU solo bridge smoke and environment dependency checks passed. A full telecom GPU training run was not repeated.

For installation, `scripts/setup.sh` uses `requirements-gpu.txt`; `requirements-local.lock` records the tested environment. The migration used the existing pinned Torch/Transformers/TRL/PEFT stack rather than replacing it.
