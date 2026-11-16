# Interactive retail RL harness

The retail path lives in `tau2_grpo.rl`. It uses real Tau2 retail Gym episodes,
with `solo_mode=False`, a frozen OpenAI-compatible customer, a frozen external
summarizer, and Tau2's terminal task evaluation. The default external endpoint is
the existing Qwen3.5-4B server on MIG2. Actor inference/training defaults to MIG0.
The launcher refuses to run while the recorded SFT process is still training.
It never stops servers or reallocates GPUs.

## Commands

```bash
scripts/run_retail_rl.sh check
scripts/run_retail_rl.sh compare --task-id 59
scripts/run_retail_rl.sh collect --task-id 59
scripts/run_retail_rl.sh train
```

`check` validates metadata, the completed SFT adapter, task partitions, context
limits and external model availability without loading actor weights or starting
training. `compare` runs one complete episode for each of the function-calling
base and retail SFT adapter, with identical task, seed, sampling parameters,
customer, judge and summarizer settings. `collect` samples one group without an
optimizer. `train` runs the configured number of on-policy rollout groups.
Training is not started by `check`, `compare` or `collect`.

The same CLI is available as:

```bash
.venv/bin/python -m tau2_grpo.cli retail-rl check
.venv/bin/python -m tau2_grpo.rl train --config configs/rl-retail.json
```

Use `--set KEY=JSON` to override strict JSON config fields, for example
`--set train_steps=2`. Choose `--partition validation` or `test` for evaluation;
training refuses either partition. The default manifest reserves 22 RL tasks,
separate from SFT, validation, official test and quarantined tasks.

## Context and replay

The actor has 8,192 total tokens, a 7,680-token maximum prompt and a 512-token
response reserve. Counts include the actual Aya chat template, policy, tool
schemas, memory and observations. The complete raw environment history is
retained independently of the actor's current context.

Before an overflowing action, a frozen summarizer compresses past history. The
policy/tool catalog stays unchanged. Memory is placed in the same user-role
`<memory>` wrapper used by SFT; the most recent complete assistant/incoming
exchange stays verbatim. Repeated compactions consume previous validated memory,
its original recent exchange, and new messages, keeping original citation IDs.
Only past history is supplied: no future action, reward rubric or reference
solution. The environment is not reset at a context boundary.

Memory provenance checks are shared with SFT in `retail_memory.py`. They reject
future citations, unsupported identities/numbers, item-field mismatches, order
status mismatches and invented authentication. Null/unknown records may have
empty citations. The controller maps a call citation only to its matching adjacent tool result,
then checks the unchanged fact values; it never searches for a convenient
observation. Authentication is derived from successful identity-lookup results,
never inferred from a profile read. These corrections are audited alongside raw
summary responses. Checks cannot prove completeness or arbitrary natural-language
faithfulness. Summary retries are bounded. If the actor context or frozen
summarizer context cannot fit, the episode is saved as an error, not silently
truncated. The teacher also has a finite 16K context.

Each action saves exact rendered prompt text, prompt IDs, generated IDs,
behavior-policy version, temperature, actual sampled-token log probabilities,
old-policy replay probabilities and, for training, frozen-reference probabilities. Replay uses those exact token arrays;
it never regenerates summaries or reconstructs a flattened conversation. Actor
text, including malformed calls, is stored verbatim. Valid tagged tool calls are
schema-validated before Tau2 execution; invalid calls get explicit runtime
feedback. Tool observations retain their identity and become Aya's user-role
`<tool_response>` messages only in the model view.

## Objective and orchestration

This is a custom turn-aware PyTorch/PEFT optimizer, rather than the legacy
telecom AgentTune/TRL trainer. Their existing single-prompt/completion contract
cannot represent changed per-action histories faithfully. The legacy telecom
entry point and the AgentTune SFT backend are unchanged. This harness does not
claim to use AgentTune's GRPO training loop.

Weights remain fixed while collecting all independent episodes of one task.
Tau2 supplies one terminal reward per episode. Group advantages use population
standard deviation plus `1e-8`; all-tie groups skip the update. An infrastructure
failure has a null reward and causes the entire group to skip optimization.
Failure rates are logged; skipping failed groups can select which trajectories
are trained on, so this is not an unbiased full-trajectory policy-gradient claim.

For each generated token, the objective uses a clipped new/behavior probability
ratio and the nonnegative `exp(ref-new) - (ref-new) - 1` KL estimator. The frozen
reference is a second, unchanged copy of the initial retail SFT adapter. Actor
LoRA dropout is disabled for consistent rollout/replay distributions. Sampling
uses temperature 0.8 with top-k and top-p filtering disabled, so replay has the
same support and normalization. Before the first update, differentiable replay probabilities must match the saved
old-policy replay probabilities within the configured tolerance. Cached BF16
generation can differ numerically from full-context replay. Its actual sampled
probabilities are retained separately: a detached correction
`exp(old_replay - sampled_behavior)` multiplies the clipped policy-change loss
and KL term. Without clipping, the product is exactly the token ratio
`exp(new_replay - sampled_behavior)`. The sampling/replay gap is logged, not
misreported as a weight change. This is token-level correction, not a claim of
an unbiased full-trajectory importance estimator.

Loss is averaged over generated tokens within each whole episode, then averaged
over episodes in the group. Splitting an episode into more actions/windows does
not give it more total weight. Summaries, customer messages, policy/tool schemas,
observations and repeated context receive no policy loss. Logits are projected
only for generated positions in small chunks; gradient checkpointing reduces
replay memory. Full-size GPU optimizer memory has not yet been validated.

Defaults: 4 episodes/group, 10 groups, 1 update/group, AdamW learning rate `5e-6`,
clipping epsilon 0.2, KL beta 0.02, gradient clipping 1.0, BF16, maximum 40 actor
actions and 16,384 generated tokens per episode. These episode limits are
separate from the per-action 8K context limit. W&B online logging is enabled in
project `tau2-retail`. There is no optimizer-resume implementation yet.

## Outputs and verification

Each run writes config/runtime metadata, JSON and readable text for every
episode, raw customer/judge replies, all summary attempts including rejected ones, action prompts
and probabilities, terminal simulation and reward. Training additionally writes
group metrics, policy adapter checkpoints and optimizer snapshots. A comparison
writes `comparison.json`; raw episodes remain available even on errors.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_retail_rl.py' -v
```

Tests force five compactions while checking one reset/reward and complete action
coverage; check context reserves, malformed-call recovery, failure handling,
task/policy group identity and segmentation-independent weighting; compare
sampled and replay probabilities on a tiny real Cohere2 model; perform a real
LoRA GRPO optimizer step while keeping its reference adapter frozen; and execute
real Tau2 reset/tool/terminal-reward integration with an explicitly stubbed local
customer/judge and no network. These do not establish improved task reward.

## Executed local validation, 2026-11-10

All 49 repository tests passed. The completed retail SFT step-46 adapter was
compared against its function-calling base on reserved RL task 59, seed 42.
Both received actual Tau2 reward 0. The base made one malformed call and no
valid tagged calls; retail produced 12 syntactically valid tagged calls but did
not execute the requested cancellation/address update. Retail continued through
one accepted compaction with a 6,246-token prompt. This one task is a wiring and
behavior diagnostic, not evidence of a reward improvement.

The completed comparison is in `outputs/retail-rl/base-vs-retail-sft-46-v2/`.
The `.conversation.txt` files place model outputs first and policy/tools in an
appendix; the retail file marks the memory boundary. The first attempt remains
separately saved because a rejected summary caused an infrastructure failure.

Replaying all 24 saved actions on MIG0 matched their stored **replay-policy**
probabilities exactly, including actions after compaction. The separate actual
sampling/replay differences were retained; their maximum absolute log-probability
gap was approximately 0.277. The explicit detached sampling correction is needed
for these BF16 cached-generation differences. A full-size LoRA backward check on
7,361 prompt plus 393 completion tokens had finite, nonzero gradients, zero
replay mismatch, and peak allocated GPU memory 9.62 GiB. It performed no optimizer
step. The GPU audit is `replay-audit.json`; no full retail RL training run was
started. MIG2 remained available throughout.

```bash
CUDA_VISIBLE_DEVICES=MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04 \
  .venv/bin/python scripts/check_retail_rl_replay.py \
  outputs/retail-rl/base-vs-retail-sft-46-v2 --all-actions --grad-check
```

Paired comparisons now automatically write `comparison-checks.txt`, Markdown,
CSV and JSON versions, plus a per-model `.checks.json`. Each recorded DB,
expected-action, environment, language-assertion and communication check retains
its individual pass/fail status and whether its category gates the task reward.
Missing checks remain unevaluated. Actor turns include every generation attempt;
total dialogue turns count actor plus customer messages, with tool responses and
summary events listed separately. Evaluation summaries are also prepended to the
readable conversation files. Existing saved pairs can be backfilled with
`tau2_grpo.rl_reporting.write_comparison_checks(run_directory)` without running
inference again.

For SFT-only observed pass@4 on the same task with four independently seeded
rollouts at temperature 0.4:

```bash
scripts/run_retail_rl.sh collect --task-id 59 \
  --set group_size=4 --set temperature=0.4
```

This does not load/evaluate the function-calling base and does not optimize the
actor. `pass-at-k.json` and `.txt` record all four rewards, individual checks and
turn counts. Observed pass@4 is true when any sampled episode has reward 1; it is
false only when all four were evaluated without success. An incomplete group
with no success remains unknown. This single-task diagnostic is distinct from
an average pass@4 benchmark across tasks. Each episode also has a readable
`.conversation.txt`; recreate readable files with
`.venv/bin/python scripts/export_retail_replays.py RUN_DIRECTORY`.

For a single greedy SFT evaluation, use `collect --task-id 59 --set group_size=1
--set temperature=0`. Temperature zero switches generation to `do_sample=False`;
model probabilities are scored at ordinary temperature 1, avoiding division by
zero. The actual deterministic chosen-token behavior probabilities are logged
separately from raw model probabilities. Greedy episodes are inference-only and
are rejected by the GRPO optimizer. Repeating greedy actor draws is not the same
stochastic pass@4 experiment. A live readable-file exporter is available as
`scripts/watch_retail_replays.py RUN_DIRECTORY --pid ACTOR_PID`.

The task-59 rerun with teacher sampling parameters (`temperature=0.7`,
`top_p=0.95`, 512 output tokens/action) also received reward 0 after 40 actor
turns. Its files are in
`outputs/retail-rl/sft-task59-teacher-decoding-temp07-topp095/`.
Nucleus sampling is supported for inference; GRPO training requires top_p=1
because the implemented replay objective assumes untruncated policy support.

A subsequent greedy diagnostic compared the base and retail adapter on three
exact training prompts from reward-1 source traces. The retail adapter reproduced
the email lookup and cancellation calls exactly, but guessed `user@example.com`
when the demonstrated response asked the customer for their email. Gold-token
NLL improved from 2.331 to 0.029 for the lookup, 1.129 to 0.0066 for cancellation,
and 1.156 to 0.811 for clarification. This small diagnostic demonstrates learning
on those two supplied-argument examples and an unresolved clarification failure
even on a seen example; it does not establish general tool-use competence or
isolate a single cause in training. Inputs and generated outputs are saved in
`outputs/retail-rl/sft-seen-prompt-diagnostic-v2/`. The script is
`scripts/diagnose_retail_sft_learning.py`; it performs no environment or optimizer
updates.

To evaluate only the earlier function-calling SFT checkpoint (merged step 525),
without loading the retail adapter:

```bash
scripts/run_retail_rl.sh collect --actor function_calling_base \
  --task-id 113 --set group_size=1 --set temperature=0
```

`--actor retail_sft` is the default. Actor selection is recorded in the run
config/runtime and report. The function-calling-only option is inference-only;
retail RL training still starts from the configured retail SFT adapter.
