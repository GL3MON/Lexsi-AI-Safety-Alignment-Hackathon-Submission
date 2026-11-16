# Building a Retail Tool-Calling Agent with Tiny Aya

## Staged function-calling adaptation, synthetic recovery trajectories, and additive reinforcement learning

This project investigates how a compact conversational model can become a reliable retail agent through staged adaptation. Starting from [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), we developed a sequence of function-calling training stages, followed by a domain-specific synthetic curriculum covering both successful workflows and recovery from failed tool calls.

The central objective is grounded task execution: asking for missing information, copying arguments accurately, choosing an appropriate operation, interpreting tool results, and completing the customer's request. Producing a plausible tool-call string is only one part of that objective.

An earlier retail curriculum contained **700 multi-turn conversations: 500 clean trajectories and 200 recovery trajectories**, all within an 8,192-token training context. Subsequent adaptation included Salesforce retail traces and focused clarification data. We then tested additive reinforcement learning on complete Tau2 tasks and a shorter, verifiable retail curriculum. The first short pilot improved held-out mini-scenario success from **1/9 to 4/9**. Later clarification-reply RL improved a separate paired development evaluation from **86/114 to 89/114**, while both compared checkpoints scored **0/3** on the selected native Tau2 tasks. Local decision gains therefore remain distinct from complete-task transfer.

This write-up records the training approach, observed checkpoint failure modes, and the first long- and short-horizon retail RL experiments. The short-curriculum results are development measurements, not a measured improvement on the native Tau2 benchmark. Earlier sections retain the details of the original 700-conversation intervention; they should not be read as the configuration or initialization of every later experiment.

## 1. Behavioral analysis before domain adaptation

Initial qualitative probes examined more than whether the model could emit JSON. We tested missing-information requests, exact copying, corrections, tool failures, and multi-step retail interactions. These probes identified several behaviors to address:

- Calling a function before obtaining its required inputs.
- Inventing email addresses, customer identifiers, order identifiers, or other argument values.
- Producing malformed JSON, unsupported function names, or incorrect argument keys.
- Repeating a failed call without using the error to revise the next action.
- Choosing an operation that did not match the order's state or the customer's request.
- Reporting completion before a tool confirmed that the requested action succeeded.
- Losing track of constraints over longer interactions.

These observations motivated a curriculum combining general function-calling instruction with retail-specific execution and recovery. They are qualitative findings from development probes, rather than a population-level estimate of the base model's error rate.

## 2. Training lineage

| Stage | Training material | Purpose |
|---|---|---|
| Starting model | Tiny Aya Global | Conversational foundation |
| Hermes SFT | Single-turn and multi-turn function-calling examples | Learn tool schemas, structured calls, and tool-conditioned dialogue |
| Glaive SFT | Prepared Glaive function-calling data | Strengthen argument selection, copying, and clarification behavior |
| Glaive RL | Interactive episodes from the prepared Glaive RL split | Optimize behavior using verifiable tool-use signals |
| Retail synthetic SFT | 700 human-designed synthetic scenarios | Adapt to retail workflows, calculator use, and explicit error recovery |
| Complete-task Tau2 retail RL pilot | Fresh environment rollouts on the fixed RL task partition | Test additive task and grounding rewards; observed group had no native successes |
| Later focused SFT initialization | Salesforce retail adaptation followed by focused clarification examples | Establish the checkpoint-25 starting policy used in both recent RL experiments |
| Short retail RL pilot | Fresh 1–4-action episodes with executable short goals | Improve local tool decisions; held-out mini-scenario success increased from 1/9 to 4/9 |
| Expanded short curriculum | 3,000 scenarios across 30 families | Broaden verifiable goal coverage and diagnose clarification failures |
| Controlled clarification SFT50 | 40 matched next-decision training examples | Learn asking versus calling; test identity transfer independently of workflow completion |
| Clarification-reply RL | Verified after-reply states plus opening and retention cases | Improve use of newly supplied facts; paired development score 86/114 → 89/114 |
| Native Tau2 curriculum RL | Original reserved RL tasks, progressively introduced | Optimize additive task checks and failure penalties without a standalone successful-call bonus |

The initial function-calling preparation used [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), including its single-turn, multi-turn, and packaged Glaive material. Hermes and Glaive therefore describe the training stages and source families; they should not be interpreted as necessarily independent data collections.

The earlier 700-conversation retail SFT continued from the post-Glaive-RL policy, built on the exported Hermes-adapted Aya model. This preserves the earlier function-calling adaptation while introducing a focused retail curriculum.

## Checkpoint-level findings and the resulting interventions

The intermediate checkpoints expose different failure modes. The sequence is not a monotonic improvement curve: a model can improve at one local behavior while remaining unreliable at complete task execution. The following observations come from saved generations and replay diagnostics. Small probe sets establish concrete examples, not general benchmark rankings.

### Untouched Aya: fabricated arguments and fabricated observations

In the native-EOS task-38 replay, the base model attempted `modify_pending_order_payment` with an unobserved order ID, `#W0000000`, and an invented payment-method ID. Within the same assistant generation, it also wrote a supposed successful tool response and a completion message. The actual environment returned `Error: Order not found`.

This revealed two distinct learning needs: separating the assistant's role from the environment's role, and grounding every action in observed facts. A plausible conversation continuation cannot substitute for an executed operation.

**Intervention:** Hermes function-calling SFT establishes explicit call boundaries and tool-conditioned responses. Subsequent retail SDG stores observations as tool messages and executes calls against a real scenario state. Assistant-generated text is excluded from the evidence used by the planned RL argument-grounding reward. Final database and action checks provide outcome supervision independently of the assistant's claims.

**Status:** These mechanisms address the observed behaviors, but the base example alone does not establish the measured effect of each intervention.

### Hermes-adapted checkpoint: recognizable calls without reliable workflow progression

The saved Hermes comparison shows that the model could call `find_user_id_by_name_zip` correctly when the customer's name and ZIP were already available. However, when no identity information was supplied, it invented `John`, `Doe`, and `12345`. In another probe, it requested a missing ZIP correctly, then asked for the first and last names again even though both were already present.

After authentication returned `jonas_keller_eval_72`, the model described the identifier as `JonasKellerEval72` and asked about return items instead of retrieving the account and order information. In the failed-email recovery probe, it described the next lookup in prose without executing it.

**Intervention:** The short retail curriculum supplies missing-field clarification, exact use of supplied identity details, alternative lookup after failure, and the transition from authentication to `get_user_details`. The required behavior is to ask only for missing information and then perform the next supported action.

**Status:** Later short-sample probes demonstrate some of these transitions, but copying and completion errors remained.

### Initial retail SFT: more tool activity without reliable completion

In the saved task-59 comparison, the retail-adapted model retrieved order `#W2702727` and correctly communicated its $625.60 total. That was useful partial progress. However, it repeatedly fetched the same order, did not execute the required cancellation or address update, and failed the final database check. Both the function-calling baseline and the retail SFT replay received reward zero.

The retail transcript also mixed observed information with unsupported details, including a quantity of two yoga mats that the tool response did not establish. Its final repeated outputs included `tool_call Successful: 1` rather than a valid requested operation.

**Intervention:** The revised curriculum connects discovery, confirmation, mutation, and final reporting in complete workflows. Multiple-goal cases require satisfying each goal, and execution checks distinguish successful reads from successful completion. The planned RL bonus credits each distinct successful grounded call only once, reducing the incentive to repeat a read merely to accumulate tool credit.

**Status:** Additional calls and a correct price statement were insufficient to solve the task. The revised curriculum is a response to that gap, not evidence that it has already closed it.

### Short retail SFT, checkpoint 350: clarification improved, copying remained fragile

On the four saved controlled scenarios, the short-2k checkpoint asked for an email when identity was missing, used supplied name-and-ZIP information, and switched lookup methods after a failed email lookup. It also asked specifically for the missing ZIP in the Maya Ellis case.

However, it changed `dalia.moreno.eval@example.org` to `daliamorenoeval@example.org` and later changed `Ellis` to `Elis`. Choosing the right function and asking the right question did not guarantee correct argument values.

**Intervention:** A targeted copy-repair curriculum adds punctuation-sensitive identifiers, leading-zero ZIP codes, corrected customer information, and exact copying from tool observations. The current SDG also varies identifiers and email forms to reduce reliance on a narrow synthetic naming pattern.

**Status:** Clarification was visibly better in these examples. Exact copying was still unresolved.

### Copy-repair SFT, checkpoint 100: selective fixes rather than complete copying reliability

The copy-repair checkpoint preserved `Maya Ellis` and ZIP `04861` in the previously problematic clarification example. It also completed the alternative-name lookup and subsequent account retrieval in the Ravi Dutta probe.

Nevertheless, it changed the same Dalia email to `dalia.moreno_eval@example.org`, replacing a period with an underscore. The accompanying claim that it would use the “exact email” did not make the actual argument exact.

**Intervention:** Argument provenance must be checked on the serialized call, independently of its accompanying explanation. The planned RL reward tests argument values against customer statements and successful tool results, with penalties for unsupported values. Correcting an error must restore the observed value, rather than produce a different plausible identifier.

**Status:** The name-copying example improved, while the email example remained wrong. This is evidence for targeted gains, not a general solution to copying.

### Long retail SFT, checkpoint 75: longer demonstrations did not preserve every clarification behavior

The long-500 checkpoint copied the Dalia email correctly and progressed from authentication to account retrieval in the supplied-identity probes. However, when Maya's ZIP was missing, it immediately called the lookup with `12345`. After the customer supplied `04861`, it invented the user ID `Maya_doe_04861` instead of obtaining an authenticated identifier from a successful lookup.

Transfer to complete tasks remained weak. In a local comparison on tasks 36, 17, and 38, all three synthetic stages received zero task reward on their three sampled episodes. These were single-attempt development measurements with a local Qwen judge, not official leaderboard scores. On task 38, the long checkpoint repeatedly changed `daikisanchez1479@example.com` into `daikis Sanchez1479@example.com` and retried the failing lookup.

**Intervention:** The new curriculum combines clean trajectories and recovery examples within the same training stage. Its recovery sequence requires an actual error followed immediately by a valid correction. The planned failure-streak penalty discourages repeated unsuccessful calls, while argument grounding prevents invented identifiers from receiving positive credit.

**Status:** Exposure to longer trajectories did not by itself establish robust long-horizon behavior. Some earlier clarification skills were absent in these probes.

### Glaive SFT, checkpoint 475, and Glaive RL, checkpoint 16: local calling skill did not establish retail competence

A saved paired diagnostic produced identical category-level results for Glaive SFT and the evaluated Glaive RL checkpoint:

| Diagnostic category | Glaive SFT 475 | Glaive RL 16 |
|---|---:|---:|
| Argument-copying probes | 5/5 | 5/5 |
| Missing-information clarification | 0/2 | 0/2 |
| Execution after confirmation | 1/2 | 1/2 |
| Connected episodes | 0/4 | 0/4 |
| Progression after authentication | 0/3 | 0/3 |

These are small, partly overlapping diagnostic categories, not an aggregate benchmark score. They show why successful isolated calls were insufficient evidence of interactive reliability. In a separate seen-training-case comparison, both checkpoints called a translation function with empty text instead of requesting the missing input. Those episodes tested clarification mechanics, not multilingual capability.

**Intervention:** The current retail SFT starts from the later post-Glaive-RL checkpoint 17 and supplies connected, tool-executed retail workflows. The next RL phase moves optimization into Tau2, where task state and required actions can distinguish a plausible local call from useful progress toward the customer's goal.

**Status:** The checkpoint-16 comparison does not demonstrate an RL improvement on those probes. It also does not prove that every Glaive RL checkpoint behaves identically.

### Hint-distilled retail checkpoints: successful short workflows alongside persistent late-turn errors

The hint-distilled step-200 checkpoint completed validation task 69 in six actor turns: authenticate, retrieve the account, inspect the order, request confirmation, cancel, and report the tool-confirmed outcome. The saved replay received reward one and passed its recorded database and action checks.

Other traces exposed persistent malformed-call loops. In the horizon audit, malformed attempts represented 5.5% of attempts at actor turns 1–5 and 33.8% at turns 21+ in the 166-episode starting-model collection. In a separate six-episode step-50 evaluation snapshot, the corresponding rates were 0% and 68.4%. These are different task collections and cannot support a matched before/after claim. They do show that later-turn syntax and recovery deserve explicit evaluation.

One failing skateboard trace repeatedly emitted an unterminated `transfer_to_human_agents` JSON argument after receiving `invalid_tool_call_json`. The first malformed call appeared before context summarization, and the audited malformed outputs ended normally rather than being cut off by the response-token limit.

**Intervention:** The 200 recovery trajectories explicitly cover malformed JSON, unknown tools, incorrect argument schemas, wrong identifiers, and inappropriate operations. Deliberately faulty calls are visible context but loss-masked; the immediate valid repair is supervised. Evaluation must measure repair success and failure streaks at different dialogue depths, rather than relying on short-task success alone.

**Status:** Task 69 demonstrates an actual successful workflow. It does not establish reliable recovery on harder or longer cases.

### Earlier replay SFT: speaker contamination required a data-level correction

An older replay-SFT task-34 output repeated the customer's first-person request as if it were the assistant's own request. Inspection of the corresponding older training data found customer-perspective messages under assistant labels, including a request beginning “I don't remember the specific order numbers…” inside a supervised assistant turn.

An adapter audit confirmed that the evaluated checkpoint was loaded and its weights had changed. A sampler-exposure audit also showed that a known contaminated training example would have been encountered before checkpoint 50. These findings establish a real source of inappropriate supervision; they do not prove that every instance of role confusion came from that example. The pre-SFT checkpoint also invented identity details under a matched first-turn prompt.

**Intervention:** The replacement generator uses explicit speaker fields, clearly labeled conversation history, deterministic perspective and customer-echo checks, and a focused whole-dialogue role review. The new dataset keeps tool observations in the tool role and separately audits the rendered Jinja template and loss masks. The current 700-sample corpus had no candidates in its stored role-echo screen.

**Status:** The replacement corrects the identified construction and validation pathway. Whether the resulting checkpoint consistently preserves roles must still be tested in generation.

### The 700-sample retail intervention and its supervision limits

That corpus combines 500 clean conversations with 200 controlled recovery conversations. It adds calculator-grounded scenarios and validates complete conversations below the context limit. These changes directly target the observed gaps in workflow execution, arithmetic, role consistency, and error repair.

Executable correctness and prose grounding remain separate. The audit flagged 117 retained conversations for unsupported card-ending claims or incorrectly copied identifiers in assistant prose, even though their tool calls replayed correctly. All 700 were retained in the selected training corpus. Those findings must not be described as resolved by successful tool replay. Filtering, correcting, or masking such prose is a possible further data intervention; it is not a completed step in the current run.

The implemented additive reward also does not independently verify every free-text claim. Its argument checks and repeated-failure penalties address calls; Tau2 communication and natural-language criteria provide task-specific language checks where available. Broader prose provenance would require additional verification.

Matched checkpoint comparisons and subsequent Tau2 experiments examine whether these demonstrations transfer to grounded execution. Successful tool replay alone does not establish reliable generated behavior; the later sections report the separate development measurements.

## 3. Human-designed synthetic data generation

The retail dataset uses **human-designed scenarios with model-generated dialogue and real tool execution**. The conversations were not manually written one by one. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) generates the dialogue within a controlled scenario framework.

Each scenario instantiates an isolated synthetic retail database with customer records, orders, products, prices, payment methods, and relevant order states. Separate customer and agent actors interact with this world. The customer actor has access to its private scenario facts; the agent must obtain information through the conversation and available tools.

Scenarios vary names, email formats, identifiers, products, prices, and combinations of customer goals. Workflows include authentication, order inspection, cancellations, returns, address changes, product comparisons, and monetary calculations. Some conversations combine multiple goals, requiring the agent to retain the customer's constraints across several tool interactions.

The intended assistant style includes brief, useful explanations alongside tool calls. Explanations should describe the next action or an observed result without claiming that an unexecuted action has already succeeded.

### Dataset composition

| Category | Conversations |
|---|---:|
| Ordinary clean retail workflows | 400 |
| Calculator-focused clean workflows | 100 |
| Recovery after malformed JSON | 40 |
| Recovery after an unknown tool name | 40 |
| Recovery after an argument-schema error | 40 |
| Recovery after an incorrect identifier | 40 |
| Recovery after an inappropriate operation | 40 |
| **Total** | **700** |

The dataset is divided into **630 training conversations and 70 validation conversations**, grouped by scenario family. Training order is randomized.

Every complete conversation is measured with the actual training tokenizer and chat template. The longest retained example contains **7,786 tokens**, below the **8,192-token** limit. Examples are not silently truncated to make them fit.

### Calculator-grounded examples

The 100 calculator scenarios comprise 31 combined-refund cases, 34 variant-price-difference cases, and 35 budget cases. They contain 120 successful calculator calls because some scenarios require more than one calculation.

The agent first retrieves relevant prices or obtains a budget from the customer. It then constructs an expression from observed monetary values, calls the calculator, and uses its returned result in the answer. Budget cases also check whether the answer correctly describes a balance or an overage.

During synthetic construction, the generation schema requires a calculator call once the necessary inputs are available. The teacher must still derive the expression; the expected answer is not supplied in its prompt. This is a dataset construction constraint, not a forced calculator rule at student inference time.

## 4. Teaching recovery through controlled replay

Successful demonstrations alone provide limited supervision for what to do after an error. The recovery curriculum explicitly supplies the sequence **failed action → observed error → corrected action → successful execution**.

For each recovery example:

1. Generate a valid conversation prefix and identify an intended tool call.
2. Verify the intended call against a cloned retail state.
3. Replace it with one controlled corruption and execute that corrupted call through the actual parser, schema checks, or retail tool.
4. Require an observable failure without an unintended database mutation.
5. Let the agent inspect the error and produce a corrected call on its next assistant turn.
6. Execute the correction and verify that it restores the intended valid action.

The saved conversation contains the **corrupted call and its actual failed response**. The original correct proposal is not inserted before the error. This avoids pairing a correct visible call with an unrelated failure response.

The five corruption families cover syntax, tool selection, argument structure, identifier grounding, and operation selection. For example, a delivered order may require a return workflow rather than a pending-order modification. These cases require using environment feedback to choose the next valid action.

### Visible context and supervised targets

Deliberately corrupted assistant calls remain visible in the conversation, but their tokens are **excluded from the SFT loss**. The subsequent correction is supervised. System instructions, customer messages, tool observations, and padding are also excluded from the loss.

For token sequence \(y\), context \(x\), and supervision mask \(m_t\), the objective is:

\[
\mathcal{L}_{\mathrm{SFT}}
= -\frac{1}{\sum_t m_t}
\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}).
\]

Here, \(m_t=1\) for supervised assistant tokens, including recovery responses, and \(m_t=0\) for excluded context and deliberately injected errors. The model therefore learns how to respond after a failure without being trained to reproduce the injected faulty call.

## 5. Conversation format and verification

The dataset stores customer, assistant, and tool messages as distinct roles. Tool observations use the canonical `tool` role. An Aya-compatible Jinja template renders these observations into the model's tool-response representation while preserving assistant explanations and structured calls.

Generation and validation apply several complementary checks:

- Speaker and perspective checks, including detection of copied customer utterances in assistant turns.
- Function-name, JSON, and argument-schema validation.
- Tool execution against the scenario database.
- Checks for authentication, ownership, confirmation, and goal-state consistency.
- Immediate correction checks for the recovery examples.
- Calculator operand provenance and numerical-result checks.
- Full-context token counts and training-label audits.

The retained corpus contains **2,996 replayed tool attempts**, including the 200 deliberately failing calls. Model-based dialogue reviews supplement deterministic checks; they are quality screens rather than formal guarantees of every natural-language claim. All 700 conversations are retained in the selected corpus. Natural-language grounding remains an explicit evaluation target alongside executable correctness.

## 6. Retail SFT configuration

The retail adaptation uses AlignTune with LoRA and assistant-only supervision.

| Setting | Value |
|---|---|
| Initialization | Post-Glaive-RL policy on the Hermes-adapted Aya model |
| Training / validation | 630 / 70 conversations |
| Epochs | 2 |
| Learning rate | \(2\times10^{-5}\) |
| Effective batch size | 8 |
| Scheduler | Cosine, 3% warmup |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Precision | BF16 |
| Maximum sequence length | 8,192 tokens |
| Validation interval | Every 50 optimizer steps |
| Checkpoint selection | Lowest validation loss among evaluated checkpoints |

Training and validation metrics are logged to Weights & Biases. Validation loss measures fit to the supervised curriculum; separate interactive evaluation is required to establish reliable task solving.

## 7. Additive Tau2 retail RL design

The complete-task RL design trains the retail-adapted policy through fresh interactions in [Tau2's retail environment](https://github.com/sierra-research/tau2-bench). The customer simulator responds to clarification questions, and tool calls execute against the environment. The reward consequently depends on what the agent actually does and what the environment confirms.

The project task manifest reserves **22 retail tasks for RL**, with 10 validation tasks and 40 official test tasks in separate partitions. These are task-level partitions; multiple rollouts of one task do not count as additional independent tasks. Development probes have included validation tasks, so validation should be treated as a development signal. Final benchmark claims require a separately controlled evaluation protocol.

The original plan was to initialize from the selected retail SFT checkpoint. The later complete-task and short-curriculum pilots instead used focused clarification SFT run `2190c37f90`, checkpoint 25, as documented in Section 8. Those experiments must not be attributed to the earlier 700-conversation checkpoint.

### Additive task criteria

The reward sums the actual evaluated Tau2 criteria available for a task:

- Final database-state reward.
- Required-action rewards.
- Environment assertions.
- Natural-language assertions.
- Required communication checks.

Each evaluated criterion contributes once. Missing categories contribute nothing. The native Tau2 outcome remains separately recorded so that the custom training reward can be distinguished from benchmark success.

### Additive outcomes and failure penalties

The current task score subtracts penalties for unsupported arguments and repeated failed calls. Successful grounded calls are audited, but do not receive a standalone positive bonus:

\[
R(\tau)=\sum_{j\in\mathcal{C}(\tau)}c_j
-0.25\frac{I}{\max(1,N)}
-\sum_{t\in\mathcal{F}(\tau)}
\min\left(1.0,\;0.25\max(0,s_t-1)\right).
\]

Here:

- \(\mathcal{C}(\tau)\) contains the evaluated task criteria.
- \(N\) is the number of eligible tool attempts.
- \(I\) counts attempts that fail argument-grounding or structural checks.
- \(\mathcal{F}(\tau)\) contains failed or invalid eligible attempts.
- \(s_t\) is the consecutive failure streak at that attempt.

Grounding uses customer-provided facts and successful tool observations. The assistant's own earlier claims are not accepted as evidence for newly invented argument values. Checks also track authentication and verified order ownership. Neither repeated nor distinct successful reads earn an independent positive bonus. Positive reward comes from evaluated task criteria. Valid completion and human-escalation calls are excluded from this tool-efficiency term.

The first failure in a streak has no additional streak penalty, allowing an isolated error followed by recovery. Subsequent failures incur penalties of 0.25, 0.50, 0.75, and then 1.0 per attempt. A valid, grounded, successful call resets the streak. The cap is per failed attempt, not per episode.

These grounding checks are conservative heuristics rather than a complete semantic oracle. Task-specific action and database checks provide the complementary signal for whether the requested work was actually performed.

### Policy optimization

The RL method uses online GRPO through the AlignTune integration. Multiple trajectories are sampled for a task, and their rewards determine relative advantages within the group:

\[
\hat A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}.
\]

The policy update uses a clipped importance-ratio objective with KL regularization to a reference policy. The initial retail configuration used four rollouts per group; the current curriculum uses sixteen. Both use temperature 0.7, a learning rate of \(5\times10^{-6}\), clipping at 0.2, KL coefficient 0.02, and at most 10 actor actions per episode.

Because the additive score can exceed one and tasks have different numbers of criteria, it will be reported alongside native Tau2 success and component-level results. A higher custom reward alone will not be presented as a higher standard benchmark success rate.

## 8. Experiments across task horizons and supervision objectives

### Complete-task experiment: no successful rollouts in the observed group

The complete-task run used the focused clarification SFT initialization, **run `2190c37f90`, checkpoint 25**. This is a distinct checkpoint from earlier retail or balanced-mixture checkpoints with the same step number. The adapter SHA-256 is `2c72a97c0abd990dd87be650ace9674051b4e30dfff8269f1e0b8a4980b0a7f2`.

In the recorded eight-rollout group for retail task **103**, every rollout received **native Tau2 reward 0**, and the mean additive training reward was **−0.634375**. The summed Tau2 criteria contributed zero; the mean grounded-call bonus was 0.05625, the incorrect-argument penalty was 0.096875, and the consecutive-failure penalty was 0.59375. The group produced one optimizer update before the experiment was stopped and redirected toward shorter scenarios.

The observed failure was behavioral: the sampled trajectories did not complete the task, and tool errors and repeated failures dominated their learning signal. Relative advantages still differed because some trajectories accumulated fewer penalties. A positive relative advantage in such a group therefore means “better than the other sampled failures,” not “successfully solved the customer's request.” This helps explain why a nonzero training gradient is insufficient evidence of useful task learning.

This was an unsuccessful initial complete-task experiment, not a controlled finding that long-horizon RL cannot work. One observed group and one update cannot establish an eventual learning curve or isolate horizon length as the sole cause. The practical response was to test whether the same starting policy could improve on shorter, fully verifiable decisions before returning to complete workflows.

### Short pilot: measurable improvement on held-out mini-scenarios

The short pilot restarted from the **same focused SFT checkpoint 25**, rather than continuing from the unsuccessful long-task update. It used real Tau2 retail tools and fixed customer replies, with **1–4 assistant actions per episode**. Eight scenario families covered email lookup, full name/ZIP lookup, missing ZIP, missing surname, missing first name, recovery after a failed email lookup, account retrieval, and authentication followed by order-status retrieval.

The reward was **+1 for completing the entire short goal and −1 otherwise**. It did not award partial credit for plausible explanations or isolated reads. Customer replies and tool observations were environment context; only the policy's generated actions participated in policy optimization. Training used fresh policy rollouts, so this was online RL rather than offline SFT.

| Pilot setting | Value |
|---|---|
| Training partition | Original 22-task retail RL partition |
| Sampled groups / rollouts per group | 16 / 8, totaling 128 training rollouts |
| Optimizer updates | 8; tied-reward groups supplied no relative update |
| Actor implementation | Transformers, BF16 |
| Sampling | Temperature 0.7, top-p 1.0 |
| Maximum assistant actions / response tokens | 4 / 128 per action |
| Context limit | 8,192 tokens |
| Learning rate / clipping / reference KL coefficient | 5 × 10⁻⁶ / 0.2 / 0.02 |
| Prompt contract | Saved Salesforce retail system prompt and retail SFT Jinja template |
| Evaluation | Same nine greedy mini-scenarios before and after training |

The evaluation combined three scenario types with reserved validation identities from tasks **69, 34, and 63**. These nine mini-scenarios were not used for policy updates or checkpoint selection.

| Held-out mini-scenario | Before RL | After RL, checkpoint 8 |
|---|---:|---:|
| Email lookup | 1/3 | 3/3 |
| Missing-ZIP clarification and lookup | 0/3 | 1/3 |
| Authentication-to-order-status | 0/3 | 0/3 |
| **Total** | **1/9 (11.1%)** | **4/9 (44.4%)** |

The gain is **three additional successes, or 33.3 percentage points**, on this small fixed development set. Email lookup generalized to all three evaluated identities, and one missing-information case improved. This provides evidence that the policy can learn useful short tool-use behavior under the verifiable reward. It does not establish reliable clarification, longer workflow completion, or native Tau2 benchmark improvement. The mini-scenarios also share underlying retail task identities, so nine should not be interpreted as nine independent full benchmark tasks.

The original order-status verifier required a specific profile-lookup intermediate step. A valid direct order lookup after authentication could consequently fail that route check. The table reports the original saved measurements unchanged; its 0/3 result should not be treated as proof that every generated status route was semantically invalid. The expanded scenario bank accepts both supported routes.

### Scaling the curriculum before another proper run

The expanded curriculum contains **3,000 short scenarios across 30 goal families and 100 synthetic customer/state worlds**. Each world contributes one case per family. The final split is **2,400 training, 300 validation, and 300 test scenarios**, with whole worlds and customer identities confined to a single partition. Product schemas and the shared catalog remain common across partitions; this tests new identities and states, not unseen tool schemas. Generation is complete: all 3,000 cases passed the final coverage gates, with 100 per family, no overlapping worlds across splits, and no duplicate public contexts. The largest checked reference prompt contains 5,740 tokens. This is a completed scenario bank, not a completed training result.

Coverage expands beyond the pilot's read-only tasks to all **15 tools advertised in the saved SFT prompt**: identity clarification, partial replies, alternative lookup after failure, account/order/product inspection, catalog listing, calculator totals and differences, cancellation, explicit confirmation and declined confirmation, account and shipping address changes, payment and item changes, returns, exchanges, human transfer, and recovery from wrong arguments, unknown tools, and malformed JSON. These are 30 explicit families, not exhaustive coverage of every possible retail exception.

Qwen3.5-4B varies a customer utterance while the driver preserves scenario facts, missing information, and approvals. Actual isolated retail tools execute the reference routes; expected customer/order state hashes check intended write outcomes. Supported alternative routes are accepted, and calculator checks allow equivalent expressions using the observed operands. Reference prompts are token-counted with the saved SFT template and a response reserve, keeping the short episodes within the 8K limit. Hidden answers, routes, and simulator state are verifier data and must never enter the actor prompt.

The expanded bank is an **RL environment curriculum**, not a collection of expert SFT targets. Its natural-language checks remain conservative heuristics and require transcript review. Final coverage, split-disjointness, duplicate-context, executable-route, and context-length gates must pass before the proper run. Pilot checkpoint 8 is the candidate initializer; no result from that future run is claimed here. The central hypothesis is that verified short decisions provide a more useful initial learning signal, after which progressively connected workflows can test transfer back to complete Tau2 tasks.

### Expanded-bank run submitted

The expanded curriculum run was submitted on November 15, 2026, starting from **short-pilot checkpoint 8**. It targets **60 balanced groups × eight rollouts (480 sampled training episodes)**, with each block of 30 groups covering every family once. This is a budget-bounded sample from the 2,400-case training pool, rather than a full pass through that pool. Settings retain Transformers BF16, temperature 0.7/top-p 1, learning rate 5 × 10⁻⁶, reference KL 0.02, and clip 0.2; responses have a 192-token reserve, with at most four actions and an 8K context. Rewards remain +1/−1 short-goal outcomes, not native benchmark scores.

Greedy development evaluation uses 30 fixed validation cases, one per family, before training, after every 30 groups, and at completion. Test cases remain untouched. The runner passed 54 tests, including all-family reference trajectories through the actual collector. The job has a 90-minute hard cap, saves the last two complete update checkpoints, and logs rollout advantages and sample tables to W&B. Submission does not establish that the target groups finished or that performance improved; those require the resulting artifacts. [Expanded-bank RL job](https://huggingface.co/jobs/GL3MON/6af9812b23985fb9cd832789).

### Expanded-bank outcome: training completed, held-out success unchanged

The expanded-bank run completed **60 groups and 480 sampled rollouts**, producing **28 optimizer updates** with **zero infrastructure errors**. Training rollouts included **242 successes and 238 failures**. Of the 60 groups, **28 had mixed rewards**, **16 were all failures**, and **16 were all successes**. The 32 tied groups supplied no relative-advantage update. Complete update checkpoints 27 and 28 were retained.

Greedy validation was **17/30 before training, 17/30 after group 30, and 17/30 at completion**. The same 17 cases passed before and after; there were no newly solved or newly failed validation cases. This larger run therefore **did not demonstrate a held-out success improvement**, despite successful sampling and nonzero policy updates. Its validation set uses one world across 30 families, so broader validation across the remaining worlds is still necessary before a generalization claim.

Failure inspection confirms persistent ungrounded arguments and missing clarification: the final policy invented `chiara.ochoa@example.com` when the email was absent and used ZIP `12345` when the ZIP was missing. It attempted cancellation before obtaining confirmation and did not acknowledge a declined cancellation in the expected next action. Failed-email recovery also supplied an unobserved ZIP rather than clarifying. On the order-status case, it authenticated and retrieved the order correctly, but its final verbose response exhausted the 192-token generation budget and was rejected as incomplete. This last failure separates a response-budget problem from incorrect tool selection; not every failed short goal reflects the same behavior.

The earlier 1/9-to-4/9 short-pilot improvement remains a valid result on that pilot's distinct mini-scenario set. It should not be generalized to the expanded curriculum, whose fixed evaluation showed no gain. Further work should examine reward/verifier validity and family-specific learning signals, especially all-failure clarification groups, and evaluate any resulting policy on additional held-out worlds before increasing the run scale.

### Clarification repair SFT with retention

Following the unchanged expanded-bank validation result, we curated **200 clarification demonstrations and 100 retention demonstrations** for one further SFT epoch, initialized from **final short-RL checkpoint 28**. The eight clarification families receive 25 training examples each: missing email identity, missing first name, missing last name, missing ZIP, three partial-reply variants, and failed-email recovery. The 100 retention examples cover 20 other families, five each, including grounded lookups, calculations, pre-confirmed writes, returns/exchanges, human transfer and error recovery. Retention is intended to reduce forgetting; preservation must still be measured.

These are constructed, executable demonstrations using the scenario bank's customer contexts and real retail tool observations, rather than unfiltered student rollouts. Assistant clarification wording varies across controlled templates. Each sequence asks for the genuinely missing field, receives the customer reply and executes the grounded lookup. All **344 train/validation records** passed execution and exact training-label audits; the maximum full sequence length is **6,051 tokens**. Only new successful assistant decisions are supervised. Earlier assistant context, failed calls, customer replies, tool results and system instructions are masked.

The additional **44 SFT validation records** use customer worlds 81–84, disjoint from the training worlds 0–79. Original fixed evaluation world80 and test worlds90–99 are excluded from SFT. These are synthetic curriculum partitions; the native RL task pool supplied the structural seeds. They should not be described as independent native-task SFT/benchmark partitions.

At the user's request, the submitted run uses **batch size 8 and gradient accumulation 2**, an effective batch of **16**, for approximately **19 optimizer steps** over 300 examples. It uses AlignTune, BF16, learning rate **1 × 10⁻⁵**, online W&B, evaluation/checkpointing every five steps, and two retained checkpoints. An earlier batch4 submission was canceled and replaced. After SFT, the selected validation-loss adapter is evaluated on the original30-case greedy development subset to test both clarification and retention. [Clarification-plus-retention SFT job](https://huggingface.co/jobs/GL3MON/6af9955a23985fb9cd833dcb). No SFT outcome is claimed at submission.

For subsequent RL, a **2:1 clarification-to-retention group sampler** is prepared. It rotates across the eight clarification families and all22 remaining families while preserving train/validation/test separation. This changes the training emphasis without eliminating the previously successful tasks. The continued run must initialize from the selected SFT adapter; it has not been launched automatically.

### Clarification SFT result and prepared continued RL

The clarification-plus-retention SFT completed all **19 optimizer steps** and selected **checkpoint19** by validation loss (0.438874). The selected adapter SHA256 is `98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811`. All504 trainable adapter tensors changed; peak reserved GPU memory was26.71GiB.

On the original fixed greedy subset, success increased **17/30 → 18/30**. The partial-ZIP case newly passed, and none of the previously passing17 cases regressed. Within the eight clarification families, this corresponds to **0/8 → 1/8**; other families remained17/22. These small-set observations support a limited targeted gain and retention on the measured cases, not broad reliable clarification. The remaining missing-information and failed-email cases still require improvement.

Continued RL is prepared from this **selected SFT19 adapter**, with **16 rollouts per group**, a target of90 groups/1,440 episodes, and two clarification groups for every retention group. Greedy validation is retained before training, every30 groups and after. The code passed47 regression/curriculum tests and the real readiness check. The continued run was subsequently submitted with **16 rollouts per group**, the **90-group target**, and a **90-minute hard timeout** at the user’s request. Its maximum cost reservation is$2.74 against the user’s stated approximately$3.50 available, retaining about$0.76. It may stop before completing every target group; saved artifacts determine actual coverage. [Continued clarification RL job](https://huggingface.co/jobs/GL3MON/6af99e2b23985fb9cd8347cf). No continued-RL outcome is claimed at submission.

### Interim clarification RL audit and redirection

The16-rollout continuation was stopped at the user's request to revisit clarification supervision. Complete checkpoints9 and10 were present at cancellation; the latest cached group snapshot contained19 groups with10 mixed-reward groups and7 all-negative groups. Useful signal existed on some cases (6/16 successful missing-email trajectories and3/16 successful missing-ZIP trajectories), while failed-email recovery had16 identical unsupported ZIP-12345 calls. These are intermediate training observations, not a final validation result.

An audit of the preceding SFT's exact decoded supervised spans counted **2,538 clarification tokens**, **13,615 tool-call/intent-note tokens**, and **60 status-text tokens**. Thus clarification questions represented only **15.7% of supervised tokens**, even though200/300 training examples were in the clarification category. Those examples also supervised the subsequent lookup, and longer calls contributed substantially more target tokens. This exposure imbalance is a plausible contributor to weak clarification learning, not proof of its sole cause. The next experiment should explicitly control target weighting, use complete-versus-missing contrast cases, and assess missing-field selection across multiple held-out customer worlds while retaining tool-use controls. No new stronger-SFT outcome is claimed.

### Explicitly weighted clarification-decision SFT

The next experiment continues from the evaluated **SFT19** adapter. The original300 conversations are transformed into **500 training records**:200 missing-input ask decisions,200 correct follow-up calls after the required answer, and100 retention conversations. SFT validation contains68 records from separate customer worlds. Ask-only examples stop at the question; the future answer is absent. Follow-up controls supervise the lookup after the supplied answer, with the earlier question masked. Historical failed calls and observations remain context-only.

To prevent longer calls from dominating the objective, this stage uses a per-example mean over supervised tokens, followed by explicit category weighting:

\[
\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}
\alpha_i\left[-\frac{1}{|T_i|}\sum_{t\in T_i}\log\pi_\theta(y_{it}\mid x_i,y_{i,<t})\right].
\]

For the500-record training set, weights are1.5 for clarification decisions,0.5 for follow-up calls, and1.0 for retention. Combined with200/200/100 examples, this gives intended objective mass **60% clarification /20% follow-up /20% retention**. This is a weighting specification, not a measured contribution to gradient norm or an accuracy claim. Non-target labels stay masked. A chunked output projection keeps the custom objective memory bounded.

The configured run uses **two epochs, batch8 × accumulation2, learning rate2×10⁻⁵**, and approximately64 optimizer steps. Evaluation and checkpointing occur every10 steps with two retained checkpoints. Matched greedy evaluation expands to **90 cases over worlds80,85,86**, all excluded from the SFT files. Those remain development measurements, not native benchmark results. Before/after comparisons and family-level retention determine whether stronger clarification supervision helps.

Twenty-nine checks passed, including dense loss/gradient parity, target-length independence, mask/collator checks, and a real tiny Cohere2+PEFT+TRL training/evaluation smoke. The training encoder verified all568 records below8K, with a maximum6051-token sequence. The [weighted-SFT job](https://huggingface.co/jobs/GL3MON/6af9a95229dd05ff0368b092) was submitted with a70-minute hard cap and a maximum$2.14 reservation, within the remaining authorized balance. It performs before/after evaluation around the SFT stage; no further RL is automatically launched. No weighted-SFT outcome is claimed at submission.

### Weighted-SFT outcome: clarification gains with excessive questioning

Training completed all **64 optimizer steps**, and validation loss selected **checkpoint60** (weighted validation loss0.481601). The saved final adapter SHA256 is `eb467853b71bbf14039b9b99bb59f6b446b95269fad0d0a7731bac7917360078`. All504 trainable adapter tensors changed. Although the provider reports a job timeout, the complete SFT result, saved adapter and full90-case post-evaluation were written, and evaluation recorded zero infrastructure errors.

| Fixed greedy development subset | Before | After |
|---|---:|---:|
| Clarification |5/24|10/24|
| Other families |52/66|45/66|
| Total |57/90|55/90|

There were12 newly passing cases and14 lost passes. World80 changed18/30→19/30, world85 stayed18/30, and world86 changed21/30→18/30. These results show that clarification behavior can improve, but the selected weighted-loss checkpoint did not improve overall task success.

The lost cases expose an excessive-questioning pattern. With a complete email already supplied, the policy asked for the customer's last name instead of calling the email lookup. With first name, last name and ZIP already present, it asked for ZIP again. After a malformed order lookup with the correct order ID still visible, it asked for the order ID instead of repairing the call. Conversely, a missing-ZIP case and a failed-email case improved: the policy asked for ZIP, received it and executed the correct lookup. Thus the intervention increased clarification behavior without reliably learning when clarification was necessary.

The weighted checkpoint should be retained as a diagnostic branch rather than promoted as an overall improvement. A subsequent decision-learning experiment needs direct supplied-information controls matched to missing-information prompts and evaluation of unnecessary questions, alongside missing-field recall and grounded execution. No further training or RL result is claimed here.

### Clarification decisions through DPO

The next experiment returns to **SFT19**, the stronger overall measured initializer (57/90), and uses decision-level preference optimization. Training contains **300 pairs**:100 missing-information cases preferring a specific clarification over an unsupported call,100 supplied-information cases preferring the grounded call over an unnecessary question, and100 retention controls. Validation contains68 pairs from separate customer worlds. Chosen and rejected candidates share the **exact same rendered prompt tokens**. Next-action preferences are verified against isolated retail tools and allowed routes. Rejected candidates are controlled reconstructions of observed failure patterns, rather than necessarily verbatim student generations.

The supplied-information controls directly add the required field to the initial customer turn where possible. Consequently the learning contrast is information availability, not merely whether a clarification already appeared in the dialogue. Failed-email controls retain the actual lookup error and supply a correction. Training uses synthetic worlds0–79; preference validation uses81–84. Worlds80,85,86 remain reserved for the90-case interactive development comparison.

The objective is standard sigmoid DPO with a frozen **SFT19 reference adapter**:

\[
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]\right).
\]

Only completion log probabilities, including the first EOS, enter the sequence scores. Prompt tokens condition both candidates and are not scored targets. Log probabilities use temperature1. The reference is the evaluated SFT checkpoint, not the adapter-disabled base model. Frozen reference scores are cached and bound to dataset and adapter hashes; they do not change during optimization. Unlike group-relative RL, these offline preference pairs can supply a contrast even when sampled student trajectories all fail.

The repository implements the objective through an explicit AlignTune trainer extension and shared memory-bounded PEFT scoring. Configuration uses **beta0.1, learning rate5×10⁻⁶, one epoch**, two pairs per microbatch and accumulation8 (effective16 pairs), for approximately19 updates. Preference validation/checkpointing occurs every five updates, retaining best and latest. Reported preference margins and accuracies are distinguished from interactive task success; the latter is evaluated on the same90 greedy cases against SFT19's saved57/90 baseline.

Twenty-one checks passed, including exact prompt/EOS replay, split/category checks, reference detachment and gradient signs, a real tiny LoRA preference update, and an end-to-end tiny AlignTune-owned optimization/checkpoint-selection test. Required checkpoints were cached locally before account migration. The [DPO run](https://huggingface.co/jobs/notGL3MON/6af9c0be23985fb9cd836b50) uses a90-minute cap on the newly funded account. No DPO improvement is claimed before its completed evaluation.

### DPO outcome: preference improvement did not yield clarification success

The DPO run completed all **19 updates** and selected **DPO checkpoint19** by validation preference loss. This is a new DPO stage initialized from **SFT19**, not the same checkpoint despite the shared step number. The final adapter SHA256 is `77aeb0a8b51ad0e8754d1483af32074104ceb86fcda545964744671f65fbfc28` and a verified copy is retained locally.

Held-out preference loss declined from approximately0.693 to **0.564191**. Reference-adjusted preference accuracy was **62/68 (91.2%)**, while raw fixed-candidate preference accuracy was **39/68 (57.4%)**. Within missing-information pairs, raw preference accuracy was only **3/24 (12.5%)**. These metrics show relative preference shifts and should not be interpreted as free-generation task accuracy.

| Same90 greedy development scenarios | SFT19 baseline | DPO19 |
|---|---:|---:|
| Clarification |5/24|1/24|
| Other families |52/66|54/66|
| Total |57/90|55/90|

The evaluation completed all90 cases with zero infrastructure errors. Three cases newly passed and five previously passing cases failed. World80 changed18/30→19/30, world85 changed18/30→17/30, and world86 changed21/30→19/30. The selected DPO policy therefore did not improve overall success and regressed on clarification.

Failure transcripts still show invented missing values: `inaya.salazar@example.com` when no email was supplied and ZIP`78901` when ZIP was absent. In another case the agent requested ZIP, received it, then narrated a lookup without emitting the call. A calculator failure used the operands in the wrong subtraction order. Improving preference against a fixed rejected answer consequently did not establish a reliable next-action policy over the full generation space.

The evaluated SFT19 model remains the stronger overall baseline among these branches. The weighted-SFT and DPO checkpoints are retained for diagnosis rather than promoted as aggregate improvements. A further experiment must measure unnecessary questions, missing-field selection, grounded arguments and execution after customer replies directly, alongside any preference-training metrics.

### Controlled SFT memorization and identity-transfer diagnostic

A smaller experiment isolated the ask-versus-call decision from full retail workflows. It used **20 matched training pairs (40 examples)**, five each for missing email, first name, last name and ZIP. Each supplied-information prompt differed from its missing-information counterpart only by the addition of that required field. A separate20 pairs used fresh customer identities and states with the same wording structure. Inputs retained the original retail system prompt and Jinja template; only the next assistant decision was supervised.

Starting from **SFT19**, training deliberately overfit the40 examples for **50 optimizer steps**, with batch4, learning rate5×10⁻⁵, and equal per-example mean NLL across asking and calling. The final checkpoint was evaluated without selecting it on held-out success. The trained adapter SHA256 is `eae5e7bc4d3d2338ac3afaa2ec9a549808a6fa57f17844f3e95d4feea11a6e1d`.

|Controlled greedy evaluation|SFT19 before|Diagnostic SFT50 after|
|---|---:|---:|
|Training examples|21/40|40/40|
|Fresh-identity examples|21/40|40/40|
|Missing-field questions, each split|1/20|20/20|
|Supplied-field calls, each split|20/20|20/20|
|Matched pairs with both halves correct, each split|1/20|20/20|

The corrected evaluation was independently audited for **80 distinct prompts per model**, exact raw/rendered case identity and absence of assistant targets from the inputs. Initial generation scores from the first diagnostic evaluator are invalid and excluded; the corrected evaluation reused the saved checkpoints without retraining. Earlier90-case weighted-SFT and DPO evaluations used90 distinct first prompts and are unaffected.

This result establishes that the model can learn the controlled decision and transfer it to new identities while retaining grounded lookup calls on these paired cases. It does not establish transfer to unseen wording, longer dialogue histories or other retail tools, because those factors were held fixed or excluded. The checkpoint is a candidate for subsequent broader evaluation and RL, not a demonstrated end-to-end benchmark improvement. [Training job](https://huggingface.co/jobs/notGL3MON/6af9cfe623985fb9cd837af7), [corrected evaluation](https://huggingface.co/jobs/notGL3MON/6af9d56223985fb9cd837e64).

### Broader90-case evaluation of diagnostic SFT50

The diagnostic checkpoint was subsequently evaluated on the **same90 retail development scenarios** used for SFT19, retaining greedy decoding,192 response tokens, four actor actions, the trained system/Jinja and unchanged route/state verification. All90 cases completed without infrastructure errors; an independent audit verified90 distinct initial prompts and exact raw-case identity.

|Metric|SFT19|Diagnostic SFT50|
|---|---:|---:|
|Correct initial clarification action|6/24|21/24|
|Complete clarification subgoal|5/24|5/24|
|Other scenarios|52/66|49/66|
|Overall|57/90|54/90|

There were five new passes and eight lost passes. **Sixteen clarification cases passed the first action and failed later**, compared with one for SFT19. A typical trace correctly requested ZIP, received it, then requested email rather than executing the account lookup. Other failures requested a last name or ZIP already supplied in earlier messages. The three failed-email scenarios still failed at the initial action.

This separates a genuine gain in initial clarification from multi-turn completion. The narrowly trained decision generalized to many first questions under other wording and dialogue prefixes, but it did not reliably teach the transition after a customer reply. The controlled40/40 result therefore cannot be treated as full retail task reliability. The observed next training target is combining previously supplied identity fields with the new reply and executing the grounded lookup. [90-case evaluation](https://huggingface.co/jobs/notGL3MON/6af9d8b429dd05ff0368e595).

### Clarification-reply RL: local progress and its limits

Starting from diagnostic SFT50, a ten-group pilot sampled sixteen trajectories per task. The curriculum mixed 60% states immediately after a customer answers a clarification, 20% original opening-clarification cases, and 20% other retail controls. The full preceding conversation remained visible; only newly sampled assistant actions received policy loss. Six groups supplied nonzero relative advantages and produced optimizer updates, while four tied groups were skipped.

On 38 greedy development cases from one held-out world, success increased from 23/38 to 30/38. After-reply lookup completion improved from 4/8 to 8/8; full opening clarification improved from 3/8 to 5/8; other retail cases improved from 16/22 to 17/22. These cases test short decisions rather than complete native Tau2 tasks.

The continuation initialized from the pilot weights with a fresh optimizer. It was stopped after **20 completed groups (320 sampled rollouts)**, producing **seven optimizer updates**; thirteen tied groups supplied no relative learning signal. The retained continuation checkpoint-7 is distinct from pilot checkpoint-6: optimizer-step numbering restarted between runs.

A paired greedy evaluation used the same 114 cases, comprising 38 scenarios across each of three held-out worlds, with identical prompts, templates and decoding settings:

| Development category | Pilot checkpoint-6 | Continuation checkpoint-7 |
|---|---:|---:|
| Use supplied information after a clarification reply | 22/24 | 24/24 |
| Complete an opening clarification scenario | 14/24 | 15/24 |
| Other retail skills | 50/66 | 50/66 |
| **Total** | **86/114** | **89/114** |

Exactly three cases changed from failure to success, with no lost passes on this set. In a representative case, both models requested a missing ZIP and received `19353`; the pilot then asked for email, while the continuation combined the existing name with the supplied ZIP and issued the correct name-and-ZIP lookup. The other changes similarly replaced unnecessary questions with grounded lookups after a last-name or ZIP reply.

The remaining failures distinguish asking from using an answer. Some cases still passed an empty missing name to a lookup, attempted cancellation before confirmation, or fetched unrelated product data rather than answering an order-status question. After-reply accuracy therefore should not be interpreted as complete clarification or workflow reliability.

The completed native Tau2 comparison used validation tasks **69, 34 and 63**, greedy actor decoding, a twenty-action budget and the same frozen Qwen customer, summarizer and NL judge. Both checkpoints received native reward zero on all three tasks. This small comparison shows no native-task improvement; it does not establish a population-level success rate. Native failure, premature termination and unevaluated checks remain distinguishable in the recorded transcripts.

### Medium workflow prototype: outcome verification versus prescribed order

A separate programmatically constructed bank contained 800 workflows: eight scenario families across 100 customer worlds, split into 640 training, 80 validation and 80 test cases. Each reference required four to seven newly generated assistant actions, starting with only the system prompt and customer request. The references exercised real retail tools, confirmation before mutations, and calculator-assisted exchanges. Reference execution and remaining-response reserves were checked against the 8,192-token context limit.

These were executable environment scenarios with scripted customer replies, not 800 independently model-generated SFT conversations. An audit also identified a limitation in the prototype verifier: it terminated a rollout when its next tool differed from a prescribed sequence, including some plausible read-tool reorderings. Consequently, its completion score confounded actual task behavior with reference-order agreement. This prototype was stopped and its intermediate weights were not used to initialize the subsequent native Tau2 run.

### Native Tau2 curriculum and successful-read reward audit

The native Tau2 experiment initializes from the continuation checkpoint-7 that scored 89/114 on the paired short-development evaluation. It uses the original reserved RL partition, an interactive frozen customer, actual retail tools and the task's evaluated outcomes, without requiring a prescribed intermediate tool sequence. The frozen Qwen model also supplies context summaries and natural-language judgments; those choices make this a development protocol rather than an official leaderboard evaluation.

An initial version retained an independent grounded-call bonus. Auditing the first sixteen-rollout group exposed a reward-design loophole: all sixteen trajectories had native reward zero and their evaluated Tau2 criteria summed to zero, yet four obtained positive shaped reward. The highest-advantage rollout received **+0.3889** solely from seven distinct grounded successful calls across nine tool attempts. It authenticated the customer and read account, order and product records but did not perform the requested jacket modification. Identical repeated calls earned no additional credit, but deduplication did not prevent bonuses for unnecessary distinct reads.

This establishes an exploitable reward signal, not proof that the model learned deliberate exploitation after one update. The corrective experiment **removes the standalone positive call bonus** while retaining summed evaluated Tau2 checks and argument/failure penalties. Re-scoring that read-only rollout under the revised objective yields zero. Task-specific action matches can still earn positive credit, whereas a plausible or successful call alone cannot. The corrected run restarts from the same pre-experiment checkpoint, rather than inheriting updates from the audited reward version.

Difficulty ordering uses a transparent metadata proxy:

\[
D(t)=A(t)+2W(t)+L(t)+E(t)+\max(0,O(t)-1),
\]

where A counts reference actions, W write actions, L natural-language assertions, E environment assertions, and O distinct referenced orders. The coefficients are heuristic. Reference actions are evaluation targets rather than a complete conversation, so this score is not a measured estimate of model difficulty.

Under the ten-actor-action cap, tasks 30 and 54 are deferred because their references already specify thirteen and twelve tool calls. The remaining twenty original RL tasks form easy, medium and hard buckets of seven, seven and six tasks. Each stage schedules ten groups; two groups in each later stage replay earlier tasks. Validation and official test tasks do not enter training.

| Corrected native Tau2 setting | Value |
|---|---|
| Sampled groups / rollouts per group | 30 / 16 |
| Initial policy and frozen reference | Pre-experiment continuation checkpoint-7 |
| Optimizer | Fresh AdamW; learning rate 5 × 10⁻⁶ |
| Sampling | Temperature 0.7; top-p 1.0 |
| Context / response budget | 8,192 / 256 tokens |
| Episode cap | 10 actor actions; 2,560 generated tokens |
| Reward | Sum evaluated task checks minus argument and consecutive-failure penalties |
| Independent grounded-call bonus | Disabled |
| Development validation | Before, every ten groups, and after; tasks 69 and 34 |

The native reward multiplies only the components selected by each task's reward basis. The additive training reward instead sums actual evaluated checks, including reference-action checks where recorded, and excludes empty-category neutral passes. Both are logged separately. Grounding and confirmation penalties remain conservative heuristics; lower penalties and higher relative advantage are not themselves evidence of solving the customer's task. Results from the corrected native run remain to be established.

### Assisted-start curriculum: reducing the remaining decision horizon

The corrected unassisted experiment was subsequently stopped in favor of an assisted-start curriculum. The objective is to make task completion reachable while retaining native Tau2 interaction and outcome checks, rather than paying for successful but potentially irrelevant reads. The initializer remains the pre-experiment continuation checkpoint-7; updates from the audited call-bonus run and the medium prototype are not inherited.

Three assistance levels progressively increase the work left to the student:

| Level | Supplied context | Newly sampled responsibility |
|---|---|---|
| Finish | Verified authentication, profile and relevant order/product observations | Select the action, obtain confirmation, execute and report the result |
| Find and finish | Verified authentication only | Retrieve the necessary information, obtain confirmation and execute |
| Unassisted | Original customer opening | Complete the full interaction |

The prefix is executed through the live Tau2 environment. Customer identity is obtained through actual customer replies; tool observations are actual results. Privileged task metadata selects relevant read operations during training, but no reference mutation call, approval or selected answer is supplied. Both environment database hashes must remain unchanged after setup. All read results remain context, and only newly sampled assistant suffixes receive policy loss. Any action-check credit earned solely by a supplied prefix is excluded from the additive training reward.

Assistance decreases only after at least 50% native task success across the latest four non-replay training groups at the current level. Later stages replay the preceding easier assistance level for 20% of groups. This is an observed training-success gate, not proof of held-out generalization. The thirty-group budget does not guarantee reaching the unassisted stage if the model does not meet the gate.

Unassisted validation remains separate, before training, every ten groups and afterward, on the fixed held-out tasks. Assisted task success is labeled as such and is not promoted as a native unassisted benchmark score. A real Tau2 integration test verified that a read-only prefix plus a scripted confirmation/mutation/report suffix can satisfy task 85 while excluding prefix actions from policy targets. That test establishes pipeline behavior, not learned model performance. Results from the assisted policy run remain to be measured. [Assisted curriculum run](https://huggingface.co/jobs/notGL3MON/6afa120923985fb9cd83af27).

### Full-trajectory SFT from the verified synthetic corpus

The next intervention returns to supervised end-to-end trajectories. The selected policy lineage includes earlier Salesforce multi-turn adaptation and multiple short-decision stages; the intervention specifically introduces the separate cleaned retail synthetic corpus, rather than assuming all prior training was single-turn.

Existing source audits identify 393 clean and 166 recovery conversations with successful real-tool replay. Their original partitions yield 499 training and 60 validation conversations. Adding 100 original clarification/retention training records and 44 corresponding validation records gives **599 training and 104 validation records**. Every combined record passed an independent rendered-token, assistant-generation-mask and installed-collator audit. Intentional failed calls remain visible context but receive no target labels. Only system prompts are standardized to the exact policy used by the rollout and RL harness; customer, assistant and tool text and selected assistant targets remain unchanged.

The demonstrated workflows include cancellation, returns, shipping-address changes, information lookup and calculator use. The corpus does not demonstrate pending-item modification or delivered exchanges, so it is not a complete retail-tool coverage intervention and should not be presented as directly teaching the jacket-change case. These covered task types are prioritized at the start of the follow-up RL task ordering.

The configured SFT uses two epochs, microbatch eight with accumulation two, learning rate 2 × 10⁻⁵, an 8,192-token limit without truncation, and equal per-example mean supervised-token loss. Clarification retention therefore contributes by examples rather than being overwhelmed solely by the longer trajectories' token counts. Validation and checkpointing occur every ten steps, with best/latest retention limited to two checkpoints. A measured longest-record batch probe precedes training.

A single unassisted native Tau2 development rollout follows the selected checkpoint. A dependency watcher then verifies model identity and a completed, scored test before launching assisted RL from that exact generated adapter. Infrastructure failures block the transition; a scored task failure remains distinct from an infrastructure failure and does not automatically cancel the explicitly scheduled experiment. No performance improvement is claimed before the resulting generations and task checks are available.

The end-to-end intervention subsequently produced a saved **checkpoint-70**, with held-out teacher-forced validation loss **0.3958**, compared with **0.3975** at checkpoint-60. Checkpoint-70 was structurally validated, cached with a content hash and selected as the initializer for the assisted RL experiment. This is a loss-based checkpoint selection, not evidence of improved generated task success. The intended standalone post-SFT rollout had not completed at selection; native smoke and unassisted baseline evaluation remain explicit gates in the subsequent RL job. [Assisted RL from SFT70](https://huggingface.co/jobs/notGL3MON/6afa811929dd05ff0369adb9).

### Experiment evidence

- [Clarification-reply pilot](https://huggingface.co/jobs/notGL3MON/6af9e60629dd05ff0368f24c): ten groups and paired one-world validation.
- [Twenty-group clarification continuation](https://huggingface.co/jobs/notGL3MON/6af9edf523985fb9cd839302): retained checkpoint-7.
- [Completed paired evaluation](https://huggingface.co/jobs/notGL3MON/6afa01a023985fb9cd83a2f8): fixed short-development cases and three native Tau2 tasks.
- [Corrected native Tau2 curriculum run](https://huggingface.co/jobs/notGL3MON/6afa0f4323985fb9cd83ac9f): task-check rewards with no independent positive call bonus.
- [Complete-task RL experiment](https://huggingface.co/jobs/GL3MON/6af8c65e29dd05ff0367c522): recorded task-103 group and additive reward components.
- [Completed short RL pilot](https://huggingface.co/jobs/GL3MON/6af9726629dd05ff03687002): 16 groups, eight optimizer updates, and before/after mini-scenario evaluation.
- Local long-run evidence: `outputs/hf-jobs/6af8c65e29dd05ff0367c522/job.log`.
- Local short-run evidence: `outputs/hf-jobs/6af9726629dd05ff03687002/bucket/short-curriculum-20261115T085045/train-f9caebd4d7/`, including `result.json`, `short-before/summary.json`, and `short-after/summary.json`.
- Expanded curriculum: `outputs/retail-data/short-scenario-bank-3000/`; its completion status is recorded in `progress.json` and the final manifest.

## 9. Evaluation priorities

Checkpoint comparisons will use fixed tasks, recorded generation settings, and readable transcripts. The evaluation will examine native task success, individual database and action checks, argument provenance, malformed calls, repeated-failure streaks, calculator accuracy, and total actor actions. Clarification behavior will be assessed on cases where required information is genuinely missing.

Particular attention will be given to recovery: whether the agent interprets an error, changes the next action appropriately, and completes the original request without inventing facts or prematurely claiming success.

The methodological progression is deliberate: general function-calling SFT establishes the interface; the retail synthetic curriculum supplies domain workflows and corrective demonstrations; environment-grounded RL then optimizes which behaviors lead to successful outcomes under interaction.

## Resources

- [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)
- [Hermes function-calling dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Tau2 benchmark](https://github.com/sierra-research/tau2-bench)
- [AlignTune](https://github.com/Lexsi-Labs/aligntune)
