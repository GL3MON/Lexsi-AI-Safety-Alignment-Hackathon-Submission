# Retail OPSD pipeline

This is the first runnable post-error privileged self-distillation path for
τ²-bench retail. It uses the existing split manifest and the configured frozen
Qwen endpoint for customer simulation and offline hindsight diagnosis.

## Training scope

The initial training pool is the manifest's **22 reserved retail RL task IDs**,
which are part of the official τ² retail train partition. The 10 validation
tasks and 40 official test tasks stay out of collection and optimization. The
40 SFT tasks are also left out of this first OPSD round, so evaluation is not
contaminated by the model's prior SFT data. Do not change the task split during
the pilot.

## Pipeline

```text
hint-free Tau2 rollout
  -> locate explicit nonterminal tool errors
  -> replay the same seed/actions and require exact prompts/observations
  -> Qwen writes diagnosis + confidence + next-turn hint after the rollout
  -> replay the same error state, try a Qwen hint, and score the recovery
  -> if it fails, diagnose that attempt and retry from the same state
     with a revised hint, up to 3 attempts per instance
  -> require a hinted recovery to reach Tau2 reward 1
  -> train on student-sampled action tokens with the successful hint
     and a frozen hinted-policy target
  -> preserve sampled actions from successful hint-free training episodes
  -> compare hint-free full tasks and matched post-error recoveries on validation
```

The initial implementation only accepts an observed tool error followed by a
later agent turn. It does not infer that a silent database mismatch is
recoverable, and it does not train from a terminal error that has no continuation
state. Those cases need explicit state verification and recovery checkpoints.
Each attempt begins from an exactly replayed original error state. If an attempt
does not reach reward `1.0`, Qwen diagnoses that completed attempt and proposes a
revised hint. OPSD retries from the same original state, up to the configurable
`max_hints_per_instance` (3 attempts by default). Each hint is appended to the
model-facing prompt for one action only. It does not enter the Tau2 conversation
or later student context. Attempts that end in infrastructure errors are not
used to generate retry hints. All diagnosis attempts, injected hints, rewards,
and rejected recoveries remain in the run artifacts.

The train loss is top-100-plus-tail symmetric JSD over the original
hint-free-student sampled tokens, with the round-start policy as the frozen
hinted teacher. Up to four sampled actions per successful training task receive
an additional same-context reference-preservation loss to bound compute. This is a pilot implementation of the proposed
method, not evidence of improvement; choose checkpoints using hint-free
full-task reward and the matched conditional recovery subset. The current
conditional recovery metric covers only explicit, nonterminal tool errors that
have a recorded next agent turn.

The Qwen diagnosis confidence is recorded as a model estimate, not a calibrated
probability. Hints that fail the exact replay check, contain detected future
identifiers, or fail to improve a matched unhinted recovery are rejected.

## Commands

Preflight check (no actor GPU model is loaded):

```bash
./scripts/run_opsd_retail.sh check
```

Run all four stages in one tracked directory:

```bash
./scripts/run_opsd_retail.sh run
```

This evaluates the initial adapter on the fixed validation split, including
matched hint-free continuations from replayable post-error states; collects 4
rollouts for each of 22 RL tasks (88 episode attempts), diagnoses up to 48
explicit-error states, validates hinted and unhinted recovery from the same
replayed state, trains for two epochs if verified improvements exist, and
evaluates the latest checkpoint on the same validation tasks and recovery
states without hints. Training stops
with an error if replayable improvements produce no accepted recovery examples.

The launcher defaults to MIG0 and refuses GPU stages while the requested
memory-holder is running. Stop the holder to release memory before launch:

```bash
kill <mig0-holder-pid>
```

You can select another visible GPU with `OPSD_GPU`, or set `OPSD_CONFIG` to a
copied config. The Qwen endpoint in `configs/rl-retail.json` must remain online.
W&B is configured online in project `tau2-retail-opsd`.

To run stages separately, create one run directory and pass it unchanged:

```bash
RUN=outputs/opsd-retail/my-run
./scripts/run_opsd_retail.sh collect --run-dir "$RUN"
./scripts/run_opsd_retail.sh diagnose --run-dir "$RUN"
./scripts/run_opsd_retail.sh validate --run-dir "$RUN"
./scripts/run_opsd_retail.sh train --run-dir "$RUN"
./scripts/run_opsd_retail.sh evaluate --run-dir "$RUN" --eval-stage after --checkpoint latest
```

Artifacts include raw rollouts under `collection/`, `candidates.json`,
`validation.json`, accepted `verified_recovery.jsonl`, AlignTune provenance,
W&B episode/intervention tables and metrics, and adapter checkpoints. Incomplete
infrastructure episodes remain unscored; they are never converted to
zero-reward examples.
