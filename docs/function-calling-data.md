# Function-calling SFT and RL data

Source: [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1),
Apache-2.0, pinned to `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`.
Only `func-calling-singleturn.json` and `func-calling.json` are used. The Glaive
and JSON-only subsets are outside this mix. Preparation does not train a model.

```bash
./scripts/prepare_sft_data.sh \
  --output outputs/function-calling-data \
  --seed 42 --rl-fraction 0.2
```

The output directory must not exist. Use a different directory for another build.
The script works from any directory, uses this project's virtual environment,
and forwards all arguments. Its default output is the project's
`outputs/function-calling-data`. Explicit relative output paths resolve against
your current directory. `make sft-split` runs the same script. The subsequent
SFT trainer carves group-disjoint training/validation sets from the SFT pool.
The installed environment supplies `transformers`, `huggingface_hub`, and
`jsonschema`; no GPU is needed. Token lengths use the pinned Tiny Aya tokenizer.

## Result

| Pool | Single-turn conversations | Multi-turn conversations | Total |
| --- | ---: | ---: | ---: |
| SFT | 841 | 711 | 1,552 |
| RL | 210 | 178 | 388 |

Single-turn means one assistant message, possibly containing several tool calls.
Multi-turn means multiple assistant messages with a recorded tool result and
follow-up; it does not imply multiple human requests. The source supplies both
short and extended versions of many initial requests. Both versions stay in the
same split even when their source IDs differ.

The seed is 42. Splitting is 80/20 by normalized initial-request group, stratified
by the turn types present and whether calls are made. Every conversation and
derived assistant decision from a group stays in that pool. No cross-pool exact
conversation, initial-request group, or assistant-prefix overlap was found.
Tool schemas can be shared across pools; semantic near-duplicates are not audited.
The RL pool is held out from SFT, not a final evaluation set.

## Files and use

- `sft.jsonl`: full conversations in `messages`, with tool schemas and provenance
  alongside. Train on assistant tokens only; mask system, user, and observations.
  Tool observation roles are mapped to `user`, preserving `<tool_response>` text,
  because Tiny Aya's native template does not support a tool role. The source
  system message already embeds the schemas: do not inject `tools` a second time.
- `rl_conversations.jsonl`: the 388 reserved full conversations, preserving canonical
  `tool` roles. These are reference trajectories, not prompts to feed wholesale.
- `rl.jsonl`: 566 next-assistant-turn examples. Feed only `prompt` to the model.
  `reward_reference` contains the separate reference response and parsed expected
  calls. Later prompts contain recorded earlier turns and tool results. This is
  offline training with recorded prefixes, not an executable multi-turn environment.
- `split_manifest.jsonl`: conversation/group IDs, split, turn type, source file/ID.
- `excluded.jsonl`: every excluded source row and the reason.
- `summary.json`: source revisions/hashes, tokenizer revision, counts, lengths,
  overlap audit, and interpretation notes.

Training/reward code is a subsequent step. Call syntax, name, and arguments can be
checked against the separate references; natural-language follow-ups require a
suitable reward and should not use literal string equality. No source APIs are
executed during preparation. A live environment would need a separate tool runtime.

## Quality filtering

Of 3,786 source rows, 1,940 are retained. Excluded:

| Reason | Source rows |
| --- | ---: |
| Invalid JSON inside assistant tool calls | 1,586 |
| Arguments violate the supplied JSON Schema | 87 |
| Invalid conversation structure, including no final assistant target | 163 |
| Tool response without a preceding call | 10 |

The builder also checks tool names, catalog consistency, response names/counts,
exact duplicates, and the 8,192-token full-conversation limit. None of the retained
conversations exceeds that limit; the maximum is 4,563 tokens. Malformed targets
are excluded, not silently repaired. Valid clarification/no-call conversations
are retained (8 SFT and 2 RL). These checks validate structure and schemas, not the
factual correctness of every generated answer or tool result in this synthetic data.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover group stability, keeping related examples together, malformed call
rejection, schema validation, tool-role conversion, and separating RL targets
from input prefixes. The build independently audits cross-pool overlap.
