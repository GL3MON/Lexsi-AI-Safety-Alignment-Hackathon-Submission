# Clarification decision DPO

Initializer and frozen reference: the evaluated SFT19 adapter, SHA256 `98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811`. The reference is the same SFT adapter, not the untouched frozen base or adapter-disabled policy.

300 training preference pairs:100 missing-input ask decisions,100 supplied-input call decisions,100 retention controls. Another68 pairs use separate validation worlds. Each pair shares one exact prompt. Chosen next actions pass the real isolated environment's reference-route check; rejected actions fail. These are decision-level preferences, not native Tau2 success labels. Rejected candidates are controlled reconstructions of observed failures rather than necessarily verbatim student outputs.

The supplied-input controls add the missing field directly to the initial customer turn whenever possible. This fixes the earlier supervision design, which emphasized calls after a clarification but gave too few direct complete-information contrasts. Failed-email controls include the authentic error followed by a supplied correction. Training only uses worlds0–79; validation uses81–84. Interactive evaluation uses80,85,86, excluded from preference training and validation.

Completion log probabilities are summed at temperature1. Prompt tokens are context, not scored targets. Both candidates are rendered through the saved SFT Jinja/rollout normalization, require an identical token prefix, and end at the first EOS inclusive. No tool/customer continuation is included in a candidate.

Standard sigmoid DPO: negative log sigmoid of beta times the chosen-minus-rejected policy log-probability difference after subtracting the frozen-reference difference. Beta0.1; LR5e-6; one epoch; two pairs per microbatch and accumulation8 (effective16 pairs), approximately19 optimizer updates. Gradients accumulate with the true pair count, including the last partial update. The output projection is memory bounded and the reference log probabilities are precomputed and bound to initializer/data hashes.

The implementation uses an explicit AlignTune trainer lifecycle with shared PEFT/PyTorch exact-context scoring. It is a repository DPO extension, not a claim that stock DPOTrainer performs our custom prompt/token replay. Core code: `src/tau2_grpo/dpo.py`, `dpo_data.py`; config: `configs/dpo-clarification.json`. Prepare data: `scripts/prepare_clarification_dpo.py`.

The job logs DPO loss, raw policy preference accuracy, reference-adjusted preference accuracy, gradient norms and validation categories to online W&B. Initial reference-adjusted margins are ties because policy equals reference; this is not a behavioral failure rate. Preference validation is every5 updates and at the end. Checkpoints include the policy adapter/tokenizer, optimizer/RNG state and metrics; keep at most two, preserving best and latest. The best validation-loss adapter receives a90-case greedy interactive evaluation against the saved57/90 SFT19 baseline. Preference accuracy is not substituted for task success.

Portable input includes local checkpoint files, source, Tau2, preference records, scenario bank and saved baseline. The frozen exported base remains gated: the new account must have access; signed-in config download is checked before a paid submission. Inputs/checkpoints live only in the new account's private bucket; no periodic Hub output commits. The old account's private bucket is not needed.

Budget: new account `notGL3MON`, $20 authorized.90-minute L40S hard cap, at most$2.74 reservation. First setup failed on gated access and was settled at a conservative$0.10 upper bound; user granted access, verified before replacement. Current replacement leaves$17.16 under the account budget. These figures are upper-bound estimates, not an invoice or independently verified wallet balance.

Read-only watcher: `.venv/bin/python -u hf_jobs/clarification_dpo/watch.py`. Current job metadata: `job.json`; source job failure metadata preserved in `job-gated-base-failed.json`. Logs and readable evaluation conversations: `outputs/hf-jobs/<job_id>/`. Live viewer runs on port8766.

21 tests passed: DPO initial loss/gradient signs, detached reference, real tiny LoRA preference update, exact prompt/EOS replay, balanced/disjoint preferences, complete tiny AlignTune-owned train/reference-cache/checkpoint-selection/final-save loop, and existing scorer/environment regressions. All required initialization/comparison checkpoints are cached under `outputs/rl-initialization/` with SHA256 provenance.
