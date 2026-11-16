# Medium-length retail RL curriculum

The current short-decision run is limited by a watcher to 20 **completed rollout groups**, not 20 optimizer updates. The watcher waits for persisted `groups.json`, written after checkpoint saving, then cancels the HF job. Tied groups count toward this limit. Any following medium run must be launched separately.

The next curriculum targets **4–8 newly generated agent actions**. A case starts with a system prompt and a customer request; there are no prefilled assistant/tool exchanges. Historical context length does not count as generated horizon. One agent action means one assistant response; tool results and customer replies are separate messages.

## Prepared workflow families

| Family | Required generated sequence | Reference actions |
|---|---|---:|
| Status with email | identity lookup → profile → order → status explanation | 4 |
| Status with missing ZIP | ask ZIP → lookup → profile → order → explanation | 5 |
| Cancel entire order | lookup → profile → order → confirmation → cancellation | 5 |
| Cancel with missing email | ask email → lookup → profile → order → confirmation → cancellation | 6 |
| Change shipping | lookup → profile → order → confirmation → address mutation | 5 |
| Change payment | lookup → profile → order → confirmation → payment mutation | 5 |
| Return delivered item | lookup → profile → order → confirmation → return mutation | 5 |
| Exchange delivered item | lookup → profile → order → product → calculate difference → confirmation → exchange | 7 |

The environment uses actual Tau2 retail tools, isolated per episode, with scripted customer replies. Confirmation and tool calls are distinct actions: asking for permission does not also execute the mutation. The reference validator executes every route and records the final database fingerprint. Policy success requires both the verified action sequence and matching final state. Tool schemas and public observations are shown to the policy; expected routes and database fingerprints remain hidden.

The medium environment allows one bounded repair opportunity for an objectively malformed JSON call, unknown tool, or schema-invalid arguments. The failed assistant call remains in the history, followed by its runtime tool error. A corrected call must still complete the same pending route. A second invalid call terminates failure. Arbitrary valid-schema business mistakes do not receive a free pass. Reference paths do not force the policy to make an error.

## Data and context checks

The preparation script builds eight workflow families over the original 100 isolated customer worlds: 640 training, 80 validation, and 80 test cases. The original 80/10/10 world separation is retained. The actual checkpoint tokenizer and SFT Jinja template measure each reference prompt. Checks reserve space under 8,192 tokens for remaining responses and a repair response. A maximum of eight agent actions bounds each episode, including repair. Reference transcripts are saved as readable `.txt` files.

## Planned training mix

- 60% full medium workflows.
- 20% the original opening-clarification scenarios, with the customer replying.
- 20% other original retail scenarios for retention.

Keep 16 on-policy rollouts per task, temperature 0.7, top-p 1.0, and exact-context policy scoring. Only sampled assistant tokens receive policy loss. Do not train on customer replies, tool outputs, or hidden reference routes. The initializer will be the retained checkpoint from the current run after its 20-group stop; record its exact hash before launching.

## Launch gate

The prepared bank and medium environment are not yet connected to the running short-bank job. Before a paid medium run, connect a dedicated mixed-bank loader, test partition isolation and loss masking, and smoke the four-to-seven-action routes with the chosen checkpoint. Check multi-turn completion, confirmation-before-write, repair behavior, context limits, and variation within 16-rollout groups. Retain separate medium completion and original clarification/retention metrics. No medium job is launched automatically by the 20-group watcher.


## Launched medium run

The mixed loader and policy-action masking checks passed before launch. The run initializes from the exact post20 checkpoint-7 hash `210d9192cfad7e5fb999d70fc2e356a8af88788e10d21865fb7d9a95dc43a983`. It targets 50 groups × 16 rollouts, 4–7 reference actions and at most 8 actions including repair, 192 response tokens, 8K context, and temperature 0.7/top-p 1.0. Three medium smoke groups test reward variance before updates; if none varies, training is skipped. The job has a 180-minute maximum reservation of $5.44.

Medium rewards now score verified progress: full database-verified completion gives +1, otherwise `min(0.99, 2 * passed_reference_steps / required_reference_steps - 1)`. A failed runtime repair attempt does not itself earn progress. Original short clarification and retention rewards stay binary. Completion rates are tracked separately from partial-progress reward.

The completed paired short evaluation of pilot vs post20 scored 86/114 vs 89/114, respectively: after-reply 22/24 vs 24/24; full clarification 14/24 vs 15/24; other retail 50/66 vs 50/66. The native Tau2 comparison required a separate resumed evaluation after a JSON presentation-wrapper parsing failure; no complete Tau2 comparison is claimed yet.
