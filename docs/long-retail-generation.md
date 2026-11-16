# Longer clarification curriculum

Generate 500 guided synthetic conversations, split across Qwen3.5-4B at localhost:8005 (MIG0) and :8004 (MIG2):

```bash
scripts/run_long_retail_generation.sh --count 500 --output outputs/retail-data/long-clarification-500
```

Ten-example smoke:

```bash
scripts/run_long_retail_generation.sh --count 10 --output outputs/retail-data/long-clarification-smoke-10
```

The driver constructs an isolated synthetic account and order for each family, using real Tau2 retail DB models and tools. Original database records are copied, original task IDs are never used, and the original DB is not mutated. Qwen rewrites dialogue slots; the driver prescribes and executes the correct tool sequence. This is a guided expert curriculum, not a measurement of free-running Qwen behavior or Tau2 benchmark reward.

Five profiles cover missing email, unavailable email with name/ZIP fallback, partial identity, one failed email lookup followed by correction, and identity already supplied. Each profile includes both cancellation and return requests. Each trace authenticates, retrieves account and order details, gets explicit confirmation, executes, and reports the tool-confirmed result. Mutation responses come from real tool execution, not generated JSON. Returns are reported as requested, not a completed refund.

For 500 examples, allocation is 50 per action/profile cell, 250 per endpoint, 450 training and 50 validation. The family split is assigned deterministically from sample index before generation. Each synthetic family belongs to exactly one split. Shared policy, products and scenario templates occur in both splits; this is not a held-out task benchmark. Smoke examples reuse production family IDs and must not be appended as additional training examples.

Artifacts:
- `records/`: immediately persisted accepted JSON and readable TXT per sample.
- `attempts/`: raw Qwen outputs and rejection reasons.
- `sft.jsonl`, `validation.jsonl`, `all.jsonl`: final exports after the whole requested batch passes.
- `conversations.txt`, `report.json`, `config.json`: review and configuration.

Rerun the same command to resume accepted records. Keep configuration unchanged when resuming. Both endpoint workers are sequential internally and run concurrently. No training is started.

Checks enforce real tool schemas, identity supplied by customer, observed account/order/item/payment identifiers, authentication and explicit confirmation before mutation, corrected failed lookup, actual final DB status and refund/return fields, assistant-only target masking, and 8192-token context. Natural-language checks are constrained substring/identifier checks and a premature-success guard; they do not prove all paraphrases semantically equivalent. Inspect the smoke TXT before scaling. All six supported tool definitions and the original retail policy are included in each system prompt; this pilot does not cover exchange or address modification.
