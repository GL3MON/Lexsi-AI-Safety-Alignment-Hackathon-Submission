# Retail replay SFT generation

Run `.venv/bin/python -u scripts/run_parallel_retail_replay_generation.py` from the repository root.
The two workers on MIG0 (port 8005) and MIG2 (port 8004) each collect 250 accepted recovery conversations. After BOTH complete, each collects 750 clean conversations. Existing accepted records are preserved.
It does not launch training. Model: Qwen3.5-4B FP8 on each GPU.

Qwen authors structured text-plus-call responses under a JSON schema, rendered into normal prose plus tool-call tags; that authoring-only instruction is excluded from training prompts. Qwen writes full agent responses and a separate customer actor receives only its private
customer facts and dialogue. Synthetic isolated RetailDB instances use randomized goals,
products, identities and order combinations; original benchmark tasks are not used.
Available tasks include cancellations, returns, address changes and order information.
These are synthetic tool-executed demonstrations, not Tau2 benchmark reward scores.

At a seeded random eligible tool turn, the intended call must first succeed on a cloned
DB. Corruptions cover broken JSON, unknown tools, argument schema, nonexistent IDs and
pending-order operations on delivered orders. Most recovery conversations contain one corruption; roughly one in seven after the pilot requires two separated corruptions. A corruption must fail on a clone without
changing its DB. The resulting real parser/schema/tool error is inserted into history.
The teacher then sees only that history, not the original correct call. Its immediately
next response must explain recovery and execute the original valid call successfully.
No hints or intended calls are exposed to the actor. Intended calls are stored in audit
metadata only. Unplanned agent failures reject the sample instead of becoming targets.

Every accepted conversation must complete its DB goals, pass an independent Qwen semantic
review (confirmations, grounded claims, final answer), and fit <=8192 tokens under the Aya
chat template including policy and tool schemas. No truncation, memory, or summarization.
The review is a model-based quality screen, not a proof of correctness. Tool results and
DB checks provide the executable checks. Recovery bad assistant spans are explicitly
masked and label-checked. Valid assistant responses alone receive loss.

`outputs/retail-data/retail-replay-2000/` contains `pipeline.log`, `progress.json`,
`pipeline-status.json`, `records/*.json` and `.txt`, all API requests/responses in `attempts/`,
rejection reasons in `rejected/`, and atomic `sft.jsonl`, `validation.jsonl`, `all.jsonl`.
The 90/10 scenario-family split reserves all five corruption types. Resume skips accepted
records; seeds and failures are recorded. A scenario exhausting 100 attempts is recorded in skipped-scenarios.json and skipped. The worker moves to a fresh scenario until the accepted-sample quota is met, without weakening checks. API/infrastructure failures remain visible errors.

If the independently generated repair is correct but its narration is vague, a separate Qwen wording pass sees only the failed call, actual error, and proposed repair. It may improve the explanation but cannot choose or alter the call. This editing request is logged in attempts; no editing instruction is included in training messages.

Worker logs: workers/mig0/generation.log and workers/mig2/generation.log. Aggregate progress: progress.json and pipeline-status.json. Training-ready JSONL contains only message/label/split fields, with no original intended-call audit metadata. In each recovery history the corrupted call is paired with its actual error, followed by the corrected call and its real success result; the clean pre-corruption proposal is never inserted. Original intended calls remain only in records/*.json audit metadata.

## Tool roles and the trained Aya Jinja

Canonical records and training JSONL store tool observations with `role: "tool"`,
a tool `name`, and the serialized JSON observation envelope in `content`. The
checkpoint's modified template is pinned at `templates/retail_aya_tool_calling.jinja`
(SHA256 in `tool-role-migration-audit.json`). It is byte-identical to the Glaive
SFT checkpoint-475 template and the exported function-calling SFT-525 template.
Its tool-role branch emits Aya's familiar turn tokens with `<tool_response>`
wrapping; this does not introduce a new special token. Customer messages remain
`user` in storage. The SFT loader preserves tool roles, and Jinja handles rendering.

Each generated record declares `chat_template_path`. The SFT trainer uses that
declaration automatically, unless an explicit config override is provided, and
saves the effective Jinja in its run and checkpoint artifacts. New generator
exports assert identical token IDs and assistant labels between the previous
tagged representation and canonical tool roles. The migration audit verified
26 saved conversations, including masking all injected bad assistant calls.
API request logs retain the teacher's historical generation inputs; they are not
the exported training dataset.

## Concurrent generation

Serving now uses `QWEN_MAX_NUM_SEQS=2` with the same FP8 model, 0.33 GPU
memory fraction, and 528-token scheduling budget on each MIG. Four independent
generation processes run: mig2 and mig2-lane1 on port 8004; mig0 and mig0-lane1
on port 8005. Their scenario ranges start at 0, 10000, 20000 and 30000. The
original workers retain 250 recovery samples each; each of the four processes
collects 375 clean samples. Total quotas remain 500 recovery plus 1500 clean.
All quality, context-length, masking, and tool-role checks remain enabled.
Worker logs live under workers/<worker-name>/generation.log. Short serving
benchmarks are saved in serving-concurrency-baseline.json and
serving-concurrency2-benchmark.json; these are not end-to-end generation ETAs.
