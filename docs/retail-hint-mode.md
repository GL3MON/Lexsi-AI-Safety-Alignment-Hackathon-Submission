# Retail HINT mode

Use `scripts/run_retail_hint.sh run` with `configs/opsd-retail-hint.json`.
The existing OPSD configs retain the recovery/retry mode.

HINT collects complete hint-free episodes, reflects on each complete trajectory,
and prepares separate teacher contexts for the original assistant turns. The
teacher receives the original pre-action context, immediate observations,
terminal reward, and turn-specific diagnosis. Both distributions are scored on
the original sampled action tokens. Failed episodes are eligible; no replay or
successful recovery is required. Diagnoses and corrections are context only.
The loss averages tokens within turns and turns within episodes. W&B logs the
original actions and diagnoses in `hint/turn_samples`.

This first training round uses a frozen round-start teacher and the existing
top-100-plus-tail JSD implementation. It does not yet reproduce the paper's EMA
teacher across multiple rounds or clipped importance weights for reused samples.
The launcher runs one collection/update round; it does not yet automate fresh
on-policy collection after each update. Do not call this a full paper reproduction.

Stages can also be invoked separately: `collect`, `prepare-hint`, and `train`,
with the same `--run-dir`. A saved collection can be prepared without rerunning
the environment, provided its initial adapter and original prompts match.

Training now accumulates configurable microbatches (`gradient_accumulation_steps`,
default 4 turns), then updates the optimizer. `eval_steps: 50` runs the full held-out
validation partition in the same process without loading another GPU model.
Evaluation is hint-free, uses fixed task/seed pairs, writes actual Tau2 episodes
and check summaries under `evaluation/step-NNNNNN/`, and logs metrics to W&B.
The final update also evaluates. `save_steps: 50` saves adapters and optimizer
state, and `save_total_limit: 2` retains the two newest numbered checkpoint
folders. Evaluation artifacts are retained independently of checkpoint pruning.
