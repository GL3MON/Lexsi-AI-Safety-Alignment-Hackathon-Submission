# Summary-conditioned SFT and RL: research and proposed first version

Status: research/design, 2026-11-10. This note does not change active collectors, rewrite the selected dataset, or start RL training. The proposed first version uses a frozen external summarizer; it does not jointly train summary generation.

## Closest papers

### ReSum: external summary tool and segmented GRPO

[ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](https://arxiv.org/html/2509.13313v1), September 2025, sections 3.1–3.3.

- A separate summary tool compresses accumulated interaction history into goal-relevant evidence and unresolved information gaps.
- After compression, the agent continues from the original query and the summary. The task continues through multiple context segments.
- The authors develop a specialized summary model through distillation/SFT, then adapt the acting agent to summary-conditioned reasoning using ReSum-GRPO.
- They compute one final reward per complete rollout, normalize rewards across the original rollout group, and broadcast that trajectory's advantage to all its segments.
- The paper's displayed objective averages across segments. Copying its advantage broadcasting does not automatically prescribe episode-balanced or token-balanced loss weighting; choose and document our weighting explicitly.

This is the closest practical starting point for our separate summarizer proposal. It was evaluated on web-search agents, not Tiny Aya retail tasks; it does not establish that our smaller Qwen summarizer will preserve every required fact.

### FoldAct: learned summaries create additional optimization problems

[FoldAct: Efficient and Stable Context Folding for Long-Horizon Search Agents](https://arxiv.org/html/2512.22733v1), December 2025, sections 3.1–3.4.

- The same trainable policy generates summaries and task actions.
- The authors identify instability from summaries changing the agent's future observations as the policy changes, plus weak summary credit assignment and high replay cost.
- They separate summary/action losses, add a consistency objective between full-history and compressed-context behavior, and train selectively on sampled segments.

Its diagnosis concerns learned, policy-dependent summaries. It is not a proof that treating memory operations as actions is mathematically invalid in every correctly specified RL formulation. Its full-history consistency loss is also not automatically feasible for our 8K student.

### CompactionRL: jointly learned compaction with PPO

[CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/html/2607.05378v1), July 2026, section 4.

- The acting policy generates summaries and continues from a summary plus a recent history tail (two steps by default).
- Summary and execution segments share a final task reward; summary generation is trainable rather than external.
- They use PPO, token-level loss normalization and a correction to advantage estimation across segment boundaries.
- They explicitly discuss the danger of counting a rollout's multiple segments as independent members of the reward-normalization group.

This is a more involved alternative if we later train the summarizer with the actor. We should not claim that freezing an external summarizer requires adopting their PPO/critic implementation.

## Proposed baseline for this repository

Use the ReSum-style external-tool flow first, with a versioned, frozen Qwen teacher endpoint. Leave its weights and summarization instructions fixed during an actor rollout/update cycle. Reuse the current teacher only after accounting for capacity used by ongoing SFT collection.

1. Keep full raw trajectories in storage. Count the student input with the actual Aya tokenizer and template, including policy/tool schemas and the memory wrapper.
2. Compact before the input reaches the 7,680-token prompt budget, which reserves 512 of 8,192 tokens for the next response. The trigger must also leave enough space for the summarization instruction if the summarizer has its own tight context limit.
3. Keep policy/tools unchanged. Summarize older completed exchanges and keep recent complete exchanges verbatim. Never split an outstanding call from its observation.
4. Use a structured state note with fields for customer identity/authentication, active requests, verified order status and ownership, exact item/product IDs and options, original/new prices, payment methods/balances, user preferences, confirmation scope, executed actions/results, unresolved errors and remaining work.
5. Attach source-message indices to summary facts. Verify identifiers and numeric values against prior customer/tool observations, including their relationships. Do not accept a number merely because it appeared somewhere in the transcript. Confirmations must identify the action/order/items they authorize. Label proposals and pending actions separately from completed actions.
6. Ensure summary plus policy/tools plus the recent tail fits the actual student budget. If not, retry within a bound or emit a preparation error with the raw trajectory preserved; do not silently discard required state.
7. For SFT, create each compacted context using only history before its target assistant message. Keep the original target unchanged; future tool results, later confirmations and reward rubrics must not enter the summary. Mask summarized/overlapping context and supervise each original target once. Keep all derived examples in their original task partition.
8. For RL, keep the same environment alive across compactions and issue one actual terminal task reward. Persist exact summary bytes, prompts/token IDs, generated tokens and behavior log probabilities. Use those exact prompts for old/new/reference calculations; do not re-summarize replayed actions.
9. Compute GRPO reward normalization across complete rollouts, then broadcast each rollout's advantage. Explicitly implement loss weighting so segmentation does not unintentionally give a rollout extra weight. Summary tokens from the frozen external model are observations, not actor actions or actor loss targets.

The frozen summarizer's conditional sampling distribution can cancel from the full trajectory importance ratio when it is identical under old/new policies and the same realized history/summary is replayed. This does not make practical clipped GRPO an unbiased full-trajectory estimator, nor does it restore a full-history agent's objective. We optimize performance of the actor operating with this summarization system.

## First experiment before dataset-wide use

Choose 3–5 saved over-8K traces, including the 11K exchange example. Construct summaries only before late target actions and check fact provenance, token fit, retained confirmation/transaction state and unchanged target coverage. Compare the teacher's next action under original versus summarized history and record failures. A summary-conditioned SFT sample can be structurally valid yet omit a prerequisite, so token fit alone is insufficient. Do not rewrite the complete selected dataset until this small experiment is reviewed.

Related deferred implementation details: [RL context-window reminder](rl-context-window-reminder.md).

## MIG0 prompt experiment, 2026-11-10

Implemented the prompt in `templates/retail_memory_summarizer.txt` and a standalone runner, `scripts/test_retail_summarizer.py`. The experiment uses the existing frozen Qwen3.5-4B FP8 server on MIG0, no environment actions, and no edits to collected training data.

The first two prompt versions preserved next-action behavior on all three examples, but repeatedly cited an older explanation for the exchange confirmation. The revised flow deliberately does not ask the model to regenerate current consent. The last original assistant explanation and customer reply remain verbatim; code attaches a `see_recent_tail` pointer with their actual indices, without asserting that consent exists. The summarizer compresses older state. This avoids trusting model-generated confirmation provenance.

Final results are in `outputs/retail-data/summarizer-test/prompt-v3-mig0/report.json`:

| Task | Aya-tokenized original pre-action prompt | Compacted pre-action prompt | Original next action reproduced |
| --- | ---: | ---: | --- |
| 2: return | 9,151 | 5,921 | Yes |
| 58: exchange | 11,466 | 6,023 | Yes |
| 23: modify pending items | 12,268 | 5,881 | Yes |

These prompt counts include policy/tools and the generation prefix, but exclude the next response. Original prompt counts render the native saved history through Aya's template; they differ from the collector's compact-JSON, full-trajectory `source_tokens` counts. All compacted inputs fit the 7,680-token input budget, reserving 512 tokens within the 8,192 total context.

All three passed critical checks for cited authentication, order ownership/status, original/replacement IDs, names, prices/options, availability, payment information, past action outcomes, pending execution status and retained confirmation-tail references. Negative unit tests check that altered prices/names/balances, future citations, omitted explanation references and fabricated execution status are detected. These checks do not prove every natural-language statement is faithful.

Current and original-context Qwen continuation calls both matched the recorded reference action on all three cases. Aya inference was not tested: only its actual tokenizer/template were used for budget measurement. The cases are selected successful traces and are too few for a general reliability estimate. The prototype currently targets late mutations with an explanation/customer-reply tail; generic cutoff handling, repeated compactions, SFT export/loss coverage and RL integration remain separate implementation work.

Reproduce with a new output directory:

```bash
.venv/bin/python scripts/test_retail_summarizer.py \
  --base-url http://127.0.0.1:8005/v1 \
  --task-ids 2,58,23 \
  --output outputs/retail-data/summarizer-test/REPLACE-WITH-NEW-DIRECTORY
```

Each case saves the actual summary request, raw model response, controller-augmented memory, compacted context, original/compacted continuation responses and validation report. Earlier v1/v2 artifacts remain for audit; the active dataset was not rewritten.
