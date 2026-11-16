# Glaive verifiable RL

Prepare the reserved RL split with:

```bash
.venv/bin/python -m scripts.prepare_glaive_rl_signals
./scripts/run_glaive_rl.sh check
```

`outputs/glaive-rl-signals/train.jsonl` contains 487 original decisions and
405 examples each of missing required information, clarification follow-up,
corrected values, and synthetic tool-result copying (2,107 total).
Variants retain the source group ID and use only the reserved RL split.
Gold responses stay in `reward_reference`; generation receives only `prompt`.

For call decisions the v1 objective is:

```
0.70 * exact call precision
+ 0.10 * tool selection
+ 0.15 * argument accuracy
+ 0.05 * literal copy accuracy
- 0.10 * extra call fraction
```

If no literal-copy argument applies, its weight uses argument accuracy instead.
Tool and argument matches use one-to-one reference allocation. Unknown/duplicate
calls cannot reuse reference credit. String case, punctuation, leading zeros,
JSON types and list order are preserved. Call sequence order is ignored; these
fixtures specify a single next call, not a parallel dependency graph.
Malformed JSON, duplicate JSON keys, nonfinite JSON constants, and fabricated
tool responses receive zero. A no-call answer on a call-required example gets
zero. Schema validity is logged separately, without a standalone positive bonus.
Partial arguments can earn partial credit even when a required argument is missing.

For missing-information examples reward is 1 if there is no call and a question or explicit request
mentions the designated missing field, else 0. This is a narrow heuristic:
it does not prove the whole response is sensible or free of unrelated claims.
Controlled fixtures explicitly list known values. Synthetic lookup tools and
observations are fixtures, not executed Glaive APIs. Free-form Qwen rewrites are
not admitted automatically, since their semantic equivalence is not verifiable
by literal substring checks. Qwen remains available for future candidate wording.

Launch collection or training explicitly:

```bash
CUDA_VISIBLE_DEVICES=MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04 ./scripts/run_glaive_rl.sh collect
CUDA_VISIBLE_DEVICES=MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04 ./scripts/run_glaive_rl.sh train
```

`configs/rl-glaive.json` selects checkpoint-475, 100 groups, 8 completions per
group, temperature 0.7, top-p 1, learning rate 5e-6 and 256 completion tokens.
AlignTune owns the training lifecycle; the shared clipped-GRPO objective replays
exact sampled tokens with a frozen reference adapter. Tied groups skip updates.
W&B online logging is enabled. JSON and TXT episodes plus reward details are
saved under `outputs/rl-glaive/run-*/group-*/`; trained adapters are saved after
actual optimizer updates. The entry point does not support resuming optimizer
state yet; starting a new run always initializes from the configured adapter.

The runner pairs each missing-required sample with its customer follow-up.
The model's actual clarification is retained in the next prompt; the reference
assistant response is never injected. The customer reveals the fixed value only
after a targeted request. A premature call ends that episode with zero reward.
After the customer reply, the model gets one final decision. Asking earns 0.25;
the final call contributes 0.75 times its call reward, for at most 1 total.
The 405 standalone follow-up rows are consumed as episode fixtures, leaving
1,702 episode starts, including 405 connected clarification episodes.
Each episode has at most two agent decisions, within the configured 10-turn cap.

Tool observations report exact reference/schema verification. Correct calls get
`call_verified: true`, `api_executed: false`; wrong calls get an error, with no
reference answer leaked. These are verifier acknowledgements, not actual API
results. A verified final call terminates the episode. No fictitious backend
result is generated, and no live Tau2 reward is used. Reliable end-to-end retail
performance still needs separate evaluation.

The system prompt includes `templates/glaive_decision_rules.txt`, which explicitly
requires clarification for missing required arguments and forbids empty values.

W&B logs `rollouts/advantage_0` through `rollouts/advantage_7` and matching
reward scalars. `rollouts/samples` is an incremental table containing every
conversation, advantage, reward, scenario, termination reason and reward details.
Episode JSON/TXT files also contain the group-normalized advantage.
