# Stronger, explicitly weighted clarification SFT

Initializer: selected previous SFT checkpoint19, SHA256 `98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811`. This is the model that scored18/30 before the canceled16-rollout RL continuation.

Training data:500 records, split from previously executed/verified demonstrations:200 missing-input ask decisions,200 supplied-answer follow-up calls,100 retention conversations. SFT validation:68 records across separate customer worlds81–84. Training uses worlds0–79. Evaluation uses worlds80,85,86, excluded from both SFT files; tests90–99 remain untouched.

Ask records end at the clarification response and have no future customer answer. Follow-up controls contain that answer and supervise only the grounded lookup; historical questions are masked. Retention keeps the previous verified calls and workflows. All568 records pass the exact assistant-span/masking checks and8K limit; max6051 tokens. Full source trajectories were tool-executed and state-hash verified. An ask-only decision is not mislabeled as completion of the entire lookup goal.

The old token-averaged objective gave clarification questions only15.7% of supervised token exposure. The new objective first averages each example's CE over its supervised tokens, then applies a fixed category weight and averages examples. Global mean weights:1.5 for ask examples,.5 for follow-up calls,1 for retention (500-record training set). Expected category loss mass is60%/20%/20%; this is objective weighting, not a claim about gradient norm or achieved accuracy. Validation weights are normalized to the same category masses. Prompt, user, tool, padding, and non-target assistant tokens stay masked.

Implementation: `src/tau2_grpo/sft_weighted.py`; opt-in `example_normalized_loss` preserves existing training defaults. The output projection is chunked and checkpointed, without full-prompt vocabulary logits. The weighted collator retains per-example weights. Trainer accumulation uses normal per-microbatch mean-loss scaling; target length does not increase an example's weight.

Config: `configs/sft-weighted-clarification.json`. AlignTune, online W&B, BF16, two epochs, batch8 × accumulation2 (effective16), LR2e-5, approximately64 optimizer steps, eval/save every10, keep2 checkpoints, select best weighted validation-loss adapter. Gradient checkpointing and MLP chunking remain enabled. No activation offloading.

The same90 greedy cases across three customer worlds run before and after training. Before-training world80 is also compared with the saved18/30 single-world result; other worlds have no older baseline until the new before-evaluation completes. The post-training evaluation compares against all90 new before cases. These are synthetic development scenarios, not native Tau2 benchmark scores.

Launch: `.venv/bin/python hf_jobs/weighted_clarification_sft/launch.py --submit`. Do not rerun while a job is active. Read-only monitor: `.venv/bin/python -u hf_jobs/weighted_clarification_sft/watch.py`. Metadata: `job.json`; results/checkpoints reside in the mounted private bucket and logs under `outputs/hf-jobs/<job_id>/`.

Budget:70-minute hard cap, maximum$2.14 reservation at currentL40S rates, within the user's remaining approximately$2.38 authorized balance after the canceled continuation; about$0.24 retained. No periodic Hub commits or automatic retries. A timeout may precede the target steps/evaluation; saved checkpoints remain available.

Verification:29 tests passed, including weighted dense-vs-chunked loss/gradient parity, target-length independence, collator masks/weights, a real tiny Cohere2+PEFT+TRL train/eval smoke with gradient accumulation, and SFT/environment regressions. The real training encoder preflight passed for all568 records before submission.
