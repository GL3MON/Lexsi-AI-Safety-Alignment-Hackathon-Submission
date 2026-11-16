# Self-Healing Tool-Use Agents via Hindsight-Guided On-Policy Distillation

**Status:** Research plan; an initial τ² retail pilot implementation is present but has not yet been run. It is not an established result.  
**Working name:** HealOPSD  
**Primary objective:** Teach an LLM agent to detect, diagnose, and recover from mistakes **after an action has already executed**, without receiving teacher hints at inference time.  
**Initial environment:** A small deterministic tool-use sandbox; extend to a stateful benchmark such as **τ²-bench (retail)** once state restoration is reliable.

---

## 1. Research question

> Can a student agent acquire an autonomous *detect → diagnose → repair → verify → resume* behavior by distilling teacher guidance at the **post-failure states the student actually reaches**?

### Core hypothesis

Hints generated from hindsight information about the student's failed trajectories can create a dense training signal even when every unassisted rollout fails. Distilling **verified, hint-conditioned recovery behavior** into the hint-free student should increase recovery success on previously failed states. Preserving the student's already-successful behavior should reduce regression.

### Distinctive angle

This work targets **recovery after a committed action** rather than only **prevention before an incorrect action**:

- **Pre-error correction:** Start from history `h_t`, before bad action `a_t`, and teach a better `a_t`.
- **Post-error recovery (ours):** Start from `h_(t+1) = (h_t, a_t, o_t)` with the action's actual effects and tool observation already present. Teach the next valid recovery action(s).
- A recovery may require compensation, rechecking state, requesting clarification, or escalation. It need not simply repeat the original action correctly.

**Scope caution:** A tool error is not always recoverable, and some failures are silent. Irreversible states and failures without observable evidence must be marked separately rather than forced into a fake recovery example.

## 2. Example: failed refund workflow

**User request:** “Cancel order #123 and refund it if eligible.”

Available tools (illustrative): `get_order_status`, `cancel_order`, `issue_refund`.

1. **Student executes an invalid action:** `issue_refund(order_id=123)`.
2. **Environment returns an observation:** `{"status": "FAILED", "error": "ORDER_NOT_CANCELLED"}`. The refund did **not** occur; the order remains active.
3. **Recovery-start state:** The entire history including the failed call and observation, plus the current environment state.
4. **Hindsight annotator (strong model) proposes a diagnosis:** “The refund requires cancellation first; verify whether cancellation is allowed, cancel, confirm, then retry the refund.”
5. **Freeze a snapshot of the student as a privileged teacher:** The teacher receives the same recovery-start state **plus** this diagnosis/hint.
6. **Test the hinted teacher continuation** in an isolated replay. Retain the hint only if the action sequence is policy-compliant and reaches the intended final state.
7. **Distill:** Student receives the recovery-start state **without** the hint; the frozen teacher supplies token/action-distribution supervision with the hint.
8. **Evaluate:** Run the updated student from the same post-failure state without teacher access. Did it recover and verify the final state?

This example has an explicit error response. A separate evaluation track must cover silent failures where the agent must first perform a verification action.

## 3. Method: high-level algorithm

```text
Initialize student policy πθ; freeze a round-start snapshot πref.

For each training round:
  1. Collect fresh, hint-free student rollouts in the sandbox.
  2. Group tasks by repeated outcome:
        all-success / mixed-success / all-failure.
  3. For each failed trajectory:
        a. Localize a candidate failure and an observable post-error state.
        b. Check that the state can be restored and recovery is possible.
        c. Retrieve related successful skill examples (optional arm).
        d. Ask a strong diagnostic model for short corrective hints.
        e. Restore the *same* post-error state and test hinted continuations.
        f. Keep only verified, non-leaking recovery interventions.
  4. Update the student using:
        a. Hint-conditioned on-policy distillation at recovery states.
        b. A preservation objective on already-successful states.
        c. Optional outcome RL on repair continuations when rewards vary.
  5. Evaluate the hint-free student on fixed held-out failure states
     AND on complete fresh tasks.
  6. Refresh rollout groups and hints as the student changes.

Stop when validation recovery and end-to-end success plateau,
or when a predefined compute budget is exhausted.
```

### Why **not** simply mix 8/8 positives and 0/8 negatives for GRPO?

A group with all rewards `1` or all rewards `0` has no group-relative reward variation. Combining such groups in a batch does not by itself produce informative GRPO advantages within either group. The proposed fix is to obtain **verified teacher-conditioned recovery targets** and an on-policy **distillation** signal. Once repair continuations include both successes and failures, outcome-based GRPO can contribute.

## 4. Dataset and environment design

### 4.1 Start with a reproducible sandbox

Build 10–20 workflow templates with stateful tools and explicit constraints (order refund, account update, return authorization, address change). Inject or naturally elicit:

- Tool argument/type mistakes.
- Prerequisite/order-of-operations mistakes.
- Incorrect state assumptions or stale reads.
- Partial completion followed by a tool error.
- Repeated failed retries.
- Wrong-target actions and policy violations.
- Silent state mismatches requiring verification.
- Irreversible actions requiring compensation or escalation.

**Use synthetic data and sandboxed tool execution, never live financial or customer accounts.**

For each task, provide a deterministic initial state, tool contracts, transition logic, allowed actions, and a **separate executable outcome verifier**. The verifier should check intended final state, forbidden side effects, and required communications, not merely whether the agent used a particular tool name.

### 4.2 Move to τ²-bench after a replay smoke test

τ²-bench is a candidate because it evaluates multi-turn customer-service tool workflows and final environment state. However, **arbitrary post-step replay/snapshotting must be implemented or verified**; do not assume benchmark infrastructure supports resuming any saved trajectory out of the box. For realistic conversations, user-simulator state and randomness must also be restored or controlled.

**Replay contract:** Given `(task_id, state_snapshot, history, random_seed)`, starting a continuation twice with the same actions must yield matching tool-visible state transitions and verifier outcomes. If this fails, do not use that sample for counterfactual repair training.

### 4.3 Rollout group definitions

For each task, sample `K = 8` hint-free student attempts initially (configurable):

| Group | Definition | Purpose |
|---|---|---|
| `P` — consistently positive | 8/8 passes | Preserve mastered skills; optional retrieval of analogous successful patterns |
| `M` — mixed | 1–7/8 passes | Ordinary outcome RL, optional repair supervision |
| `N` — consistently negative | 0/8 passes | Main source of hindsight diagnosis and guided recovery |

Do not assume every `N` task contains a recoverable state. Tag each case as **recoverable**, **needs missing information**, **requires compensation/escalation**, **irreversible**, or **undetermined**.

### 4.4 Persist a structured record

A minimal record (JSONL or Parquet) should include:

```json
{
  "task_id": "retail_001",
  "task_split": "train",
  "domain": "retail",
  "rollout_id": "retail_001_seed_2",
  "student_checkpoint": "round_00",
  "rollout_reward": 0.0,
  "failure_type": "missing_prerequisite",
  "failure_step": 3,
  "student_visible_history": "...",
  "action_at_failure": "issue_refund(order_id=123)",
  "tool_observation": "ORDER_NOT_CANCELLED",
  "post_error_snapshot_id": "snapshot_abc",
  "recoverable": true,
  "hint": "Verify and complete cancellation before refunding.",
  "retrieved_success_ids": [],
  "hinted_rollout_reward": 1.0,
  "hint_validation_passed": true,
  "hint_free_eval_reward": null
}
```

Store full structured messages/actions/observations in associated files rather than relying on the abbreviated strings above. Version the tool schemas, environment code, teacher checkpoint, hint prompt, and verification logic.

## 5. Failure detection and localization

**Goal:** Find a state from which the student can plausibly recognize and repair a mistake.

1. Run task-level verification on completed trajectories.
2. Use structured tool errors, policy rules, state deltas, and/or a diagnostic LLM to identify candidate bad steps.
3. Prefer the **earliest *observable* post-error state** where a corrective action can still change the outcome.
4. Validate that the hint-free student at that state would have access to the necessary signals. If the failure is silent, the desired next action may be a *verification tool call*, not a repair call.
5. Mark annotation uncertainty and retain multiple candidate steps if localization is ambiguous; test them independently.

**Do not leak** future trajectory details, hidden task targets, grader feedback, or ground-truth state into the student's runtime history. The diagnostic model may inspect hindsight during training, but its emitted hint must describe actionable behavior grounded in information the agent could obtain legitimately.

## 6. Hint generation, positive retrieval, and verification

### 6.1 Hint constructor

Input to an external strong model:

- User objective and tool schemas/policies.
- Failed rollout with action and observation history.
- Candidate failure location and known environment feedback.
- Optional related successful examples, used **only for hint construction**.

Required structured output:

```json
{
  "failure_mode": "missing_prerequisite",
  "diagnosis": "Refund attempted before cancellation was confirmed.",
  "hint": "Check cancellation eligibility, complete cancellation, verify it, then retry refund.",
  "evidence_steps": [3],
  "expected_recovery_check": "order_cancelled_and_refund_recorded",
  "recoverability": "recoverable"
}
```

**Hint rules:** Short, procedural, specific to the failure, no privileged answer leakage, no invented tool functionality, no instructions to bypass policies. A hint may instruct the model to verify or ask the user for information. Reject hints that presume inaccessible facts.

### 6.2 Positive bank: a controlled experimental factor

Store verified successful trajectories, indexed by **tool dependencies, workflow subgoals, action preconditions, and failure-relevant skills**; semantic query similarity alone is insufficient.

For a failed task, retrieve `m` related successes and use them to help the hint constructor infer a *generalizable repair principle*. Do **not** concatenate unrelated successful trajectories and failed trajectories as if they were preference pairs, or provide those examples to the student during hint-free evaluation.

Compare these independent conditions:

- **No retrieval:** diagnostic model sees only failed trajectory and permitted hindsight.
- **Random positive retrieval:** controls for additional context length.
- **Skill-matched retrieval:** selects examples sharing relevant preconditions and tools.

### 6.3 Hint validation

For each candidate hint `z`, restore the same post-error snapshot and generate a continuation using a **frozen student snapshot conditioned on `z`**. Apply the independent verifier.

Start with one rollout per hint; optionally retest promising hints with 2–3 different seeds. Keep only hints satisfying:

- Valid, policy-compliant actions.
- Genuine state recovery (or appropriate safe escalation), not task abandonment disguised as success.
- No unacceptable irreversible side effects.
- Verified outcome or independently verified partial recovery criterion.
- Replay consistency and hint anti-leakage checks.

Store both passing and failing hint attempts for analysis; train initially only on accepted interventions. Use teacher-hint optimization/revision only after the one-shot baseline is established.

## 7. Model training

### 7.1 Separate the roles

There are two distinct “teachers” in the proposed pipeline:

1. **Hint annotator / optimizer:** A capable external model used offline to analyze errors and propose diagnoses. It does **not** need to provide student-compatible token logits.
2. **Privileged distillation teacher:** A **frozen snapshot of the student's model** (OPSD-style), conditioned on the verified hint `z`. The live student sees the identical runtime history without `z`.

This separation matters: a stronger diagnostic model can propose useful hints, while same-architecture frozen-student logits remain suitable for self-distillation. A separate large-model-to-small-model distillation variant can be tested as a baseline but should not be mislabeled self-distillation.

### 7.2 Recovery objective

Let `s` denote a replayable **post-error history/state**, and let `z` be a verified hint.

- Student policy: `πθ(· | s)`
- Privileged teacher: `πref(· | s, z)` where `πref` is frozen for the training round.
- Sample recoveries **from the unhinted student**, not exclusively from the hinted teacher, to keep the supervision on-policy.

A conceptual on-policy reverse-KL objective is:

\[
\mathcal L_{\mathrm{recover}} =
\mathbb E_{s\sim\mathcal D_{\mathrm{recover}},\,y\sim\pi_\theta(\cdot\mid s)}
\left[\sum_{j\in\mathcal M(s,y)}
\left(\log\pi_\theta(y_j\mid s,y_{<j})
-\log\pi_{\mathrm{ref}}(y_j\mid s,z,y_{<j})\right)\right].
\]

`M` masks the relevant agent-generated repair tokens/actions. **Implementation note:** A practical sampled reverse-KL/OPSD estimator must correctly handle policy sampling, gradient flow, sequence masking, frozen teacher scores, and normalization; use a tested OPSD implementation rather than naively treating detached sampled log-prob differences as ordinary cross-entropy. Compute teacher and student log-probabilities on corresponding student-generated prefixes.

**Current retail pilot objective:** The implementation in `src/tau2_grpo/opsd_retail.py` uses symmetric top-100-plus-tail JSD at the original hint-free student token prefixes, with a frozen same-policy hinted teacher. This follows the token-distribution comparison used in the supplied HINT work and is distinct from the sampled reverse-KL equation above. Report the pilot as privileged on-policy self-distillation; do not claim it is an exact implementation of that reverse-KL estimator.

### 7.3 Preserve known successes

On a separate set of successful trajectories/states, maintain a reference-policy anchoring or supervised-preservation objective:

\[
\mathcal L_{\mathrm{preserve}}=
\mathbb E_{s^+\sim\mathcal D_{\mathrm{positive}}}
D_{\mathrm{KL}}\left(\pi_{\mathrm{ref}}(\cdot\mid s^+)\,\|\,\pi_\theta(\cdot\mid s^+)\right).
\]

This protects valid tool choice, policy adherence, and ordinary task performance. Include positive states that require **no intervention** so the student does not develop a tendency to over-correct.

### 7.4 Optional outcome optimization

Once on-policy repair continuations show varied success, add GRPO or another outcome-based method using verifier rewards **from the same recovered state**:

\[
\mathcal L_{\mathrm{total}}=
\lambda_{r}\mathcal L_{\mathrm{recover}}
+\lambda_{p}\mathcal L_{\mathrm{preserve}}
+\lambda_{o}\mathcal L_{\mathrm{outcome}}.
\]

Set `λo = 0` in the initial experiment. Increase it only after the recovery-distillation baseline works and reward variation exists. Avoid rewarding long explanations, “I fixed it” claims, or unnecessary tool calls; reward **verified final state** and penalize avoidable cost/unsafe actions.

### 7.5 Pilot training recipe

- **Model:** A manageable open instruct model with tool-calling ability; use a single model family across controlled ablations.
- **Round 0:** Baseline inference and recovery-state collection.
- **Warm-start baseline:** SFT only on verified repair continuations, to isolate the value of OPSD.
- **Main run:** Post-error hint-conditioned on-policy distillation + success preservation.
- **Rounds 1–3:** Refresh rollouts/hints, prioritize failures not yet recovered.
- **Monitoring:** Recovery success, end-to-end success, action validity, positive-task regression, teacher-hint validation rate, and compute/tool-call budget.
- **Model selection:** Pick the checkpoint with best **hint-free** held-out recovery performance subject to non-regression and safety checks.

## 8. Experiments and ablations

Maintain equal training tasks, comparable environment calls, and documented token budgets wherever possible.

| ID | Experiment | Question answered |
|---|---|---|
| B0 | Base student, no training | How often does it spontaneously recover? |
| B1 | Standard successful-trajectory SFT | Is ordinary tool training sufficient? |
| B2 | SFT on verified recovery continuations | Does simple imitation already solve recovery? |
| B3 | Generic “please reflect” at inference, no training | Can prompting alone repair failures? |
| B4 | Pre-error OPSD with hints | Are gains mainly error prevention? |
| **M1** | **Post-error OPSD with verified hints** | **Does distillation teach true recovery?** |
| M2 | M1 + positive-preservation loss | Does preservation prevent regressions? |
| M3 | M2 + skill-matched positive retrieval for hint creation | Does transfer from mastered skills improve repair? |
| M4 | M3 + on-policy outcome RL on repairs | Does outcome optimization improve verified recovery? |

Additional ablations:

- Failure type (explicit tool error vs silent incorrect state).
- Recovery position (immediately after wrong action vs a later observable state).
- Hint granularity (generic; failure diagnosis; procedural repair plan).
- Hint source (strong external annotator; self-generated; retrieved examples).
- Hint-validation filtering (verified only vs unfiltered).
- Frozen hints vs refreshed hints after each round.
- Preserving all-success data vs omitting it.
- Evaluating **with hints** (upper bound only) vs **without hints** (primary metric).

**Novelty test:** M1 versus B4 isolates recovery *after* execution from prevention *before* execution. M3 versus M2/M1 isolates whether the success bank adds value beyond hindsight hints. If neither difference is positive, narrow or revise the claim.

## 9. Evaluation protocol

### Main metrics

1. **Conditional recovery success rate:** Fraction of a **fixed held-out set of recoverable post-error states** completed successfully by the unhinted model.
2. **End-to-end task success:** Fresh full episodes from initial state; catches degradation hidden by conditional evaluation.
3. **All-fail conversion rate:** Percentage of tasks that were `0/K` with the original model but have at least one successful hint-free rollout with the trained model (report also pass@1).
4. **Recovery efficiency:** Extra tool calls, steps, tokens, and simulated execution cost to reach success.
5. **False recovery / unnecessary modification rate:** Whether the model perturbs workflows that were already correct.
6. **Policy and side-effect safety:** Invalid tools, forbidden operations, repeated damaging actions, unapproved compensation, and appropriate escalation.

### Evaluation discipline

- Split by **underlying task/template and scenario**, not by rollout, to prevent leakage.
- Freeze evaluation snapshots and verifiers before training.
- Use multiple random seeds and report confidence intervals where sample size permits.
- Report per-failure-type performance, not just aggregate success.
- Track the fraction of failed states that are genuinely recoverable; do not mix irrecoverable failures into recovery-rate denominators without separate reporting.
- Teacher hints and success-bank exemplars are unavailable to the deployed test-time student.
- If a post-error state was created by an artificial fault, disclose this and separately evaluate naturally occurring failures.

## 10. Implementation milestones

### Phase A — Environment and evaluator

- [ ] Implement a deterministic toy workflow environment with tool-state transitions.
- [ ] Build a final-state verifier and policy-violation checks.
- [ ] Implement snapshot, restore, and reproducible continuation.
- [ ] Run 20 replay consistency tests (same snapshot + same actions → same outcome).
- [ ] Establish baseline hint-free success and recovery rates.

**Exit gate:** Post-error states can be restored and independently graded.

### Phase B — Failure collection and diagnosis

- [ ] Collect `K=8` rollouts per training task.
- [ ] Partition tasks into all-success / mixed / all-failure.
- [ ] Localize candidate failure steps and save the *post-error* snapshot.
- [ ] Label recoverability and categorize failure modes.
- [ ] Audit at least 30 failure annotations manually.

**Exit gate:** Have a reproducible, labeled failure-state set whose training inputs contain only legitimate observations.

### Phase C — Hint-conditioned recovery generation

- [ ] Implement the diagnostic model prompt and structured output parser.
- [ ] Generate short repair hints for recoverable failed states.
- [ ] Replay hinted continuations under the frozen student snapshot.
- [ ] Check outcome validity, safety, and leakage.
- [ ] Save only verified intervention pairs for the first training run.
- [ ] Build an optional successful-skill retrieval index.

**Exit gate:** Produce at least a small, non-empty set of verified hints that improve frozen-model recovery compared with no hint.

### Phase D — Training

- [ ] Implement repair-continuation SFT baseline.
- [ ] Implement frozen hinted-teacher logits on unhinted student rollouts.
- [ ] Implement a tested token-level OPSD loss, repair mask, and preservation loss.
- [ ] Train baseline B2 and main variants M1/M2.
- [ ] Refresh rollouts after each iteration rather than reusing stale samples indefinitely.
- [ ] Add M3 skill-matched retrieval; add M4 outcome RL only if earlier stages work.

**Exit gate:** Demonstrate hint-free gains in held-out conditional recovery without unacceptable loss of normal task success.

### Phase E — Stateful benchmark evaluation

- [ ] Verify τ²-bench state restoration and user-simulator replay consistency.
- [ ] Port training to one domain (retail first).
- [ ] Run B0–B4 and M1–M3 with fixed splits.
- [ ] Evaluate explicit errors, silent failures, reversible faults, and escalation cases separately.
- [ ] Produce per-failure-type tables, recovery curves, compute-cost plots, and qualitative traces.

**Exit gate:** A reproducible result, positive or negative, that isolates **post-error** learning from ordinary tool-call accuracy improvements.

## 11. Suggested repository layout

```text
self_heal_opsd/
├── README.md
├── plan.md
├── configs/
│   ├── environment.yaml
│   ├── collection.yaml
│   ├── hints.yaml
│   └── train.yaml
├── environments/
│   ├── toy_workflows.py
│   ├── replay.py
│   └── tau2_adapter.py
├── data/
│   ├── schema.py
│   ├── collect_rollouts.py
│   ├── classify_failures.py
│   └── retrieve_successes.py
├── teachers/
│   ├── diagnose.py
│   ├── propose_hints.py
│   └── validate_hints.py
├── training/
│   ├── recovery_sft.py
│   ├── opsd_recovery.py
│   ├── preservation.py
│   └── outcome_rl.py
├── evaluation/
│   ├── run_recovery.py
│   ├── run_end_to_end.py
│   ├── metrics.py
│   └── analyze_failures.py
└── tests/
    ├── test_replay.py
    ├── test_hint_leakage.py
    ├── test_verifier.py
    └── test_loss_masking.py
```

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Teacher hint leaks hidden ground truth | Constrain hints to permissible observations/actions; audit hints; never give grading secrets to student |
| Training only prevents the mistake instead of teaching recovery | Start the primary loss **after** the bad action; evaluate fixed post-error states separately |
| Recovery state cannot be replayed faithfully | Snapshot full environment plus user-simulator state; reject nonreproducible interventions |
| Correctness verifier accepts unsafe shortcuts | Check policy compliance, side effects, and completion criteria independently |
| A 0/K failure is not fixable from its terminal state | Identify earlier recoverable **post-error** checkpoints; mark irrecoverable cases honestly |
| Teacher gives plausible but ineffective advice | Accept hints based on executable validation, not teacher confidence alone |
| Distillation causes catastrophic forgetting | Preserve all-success behavior and require end-to-end non-regression |
| Student only memorizes particular error strings | Hold out tools, templates, and error phrasing; test silent errors and novel state combinations |
| All repair rollouts fail, so GRPO still has zero signal | Use verified OPSD/SFT supervision first; introduce RL only after success variation emerges |
| Agent over-repairs or repeats harmful actions | Train no-op/stop/verify/escalate cases; measure false interventions |

## 13. What success would establish

A convincing result would show that, **without hints at inference**, the trained model:

1. Resolves a larger fraction of previously failed **post-error states** than SFT, generic reflection, and pre-error-only training baselines.
2. Maintains or improves end-to-end task success without elevated policy violations.
3. Generalizes beyond the exact tools and failure descriptions seen in training.
4. Benefits from verified teacher diagnoses, with a measurable additional benefit (if any) from skill-matched successful examples.

If post-error OPSD yields no advantage over SFT on corrected suffixes, that is an informative finding. The core claim should be tested rather than assumed.

---

## 14. Research grounding and related work

These papers motivate parts of the approach; **the specific full pipeline above is a proposed combination and extension, not an algorithm claimed to be implemented in any one paper**.

1. **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (OPSD)** — student-generated rollouts with a privileged-context self-teacher and token-level distillation. https://arxiv.org/abs/2601.18734
2. **Privileged On-Policy Self-Distillation (PSD)** — verified local hints for failed agent trajectories and preservation on successful examples. https://www.canvas.inc/research/privileged-self-distillation
3. **Scaling On-Policy Self-Distillation with Hint Auto-Optimization** — model-generated hints evaluated via teacher rollouts and graders, with refreshed hint optimization. https://appliedcompute.network/research/scaling-on-policy-self-distillation
4. **AHEAD: Adaptive Hindsight with Environment-Augmented Distillation for Agentic RL** — step-level error identification and corrective-hint privileged information for multi-turn agents. https://arxiv.org/abs/2608.24114
5. **AgentRefine: Enhancing Agent Generalization through Refinement Tuning** — trains agents to correct erroneous actions using environment observations. https://arxiv.org/abs/2501.01702
6. **ToolACE-R: Model-aware Iterative Training and Adaptive Refinement for Tool Learning (AAAI 2026)** — motivates separating examples by the current model's measured difficulty; this plan specifically investigates the consistently failed group it largely filters out. (Paper supplied with this project.)
7. **τ²-bench environment overview** — multi-turn customer-service tool execution and final-state evaluation. https://huggingface.co/docs/openenv/environments/tau2

### Immediate next action

**Run the small retail pipeline pilot first:** the initial implementation is documented in [`docs/opsd-retail-pipeline.md`](docs/opsd-retail-pipeline.md). It collects the reserved retail RL task split, admits only explicit post-tool-error states, requires deterministic replay and a full-reward hinted improvement over an unhinted continuation, then runs a small distillation update and compares hint-free validation before and after. It has not yet passed a Tau2 replay smoke run; treat replay parity as a hard gate.
