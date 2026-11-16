# Native Tau2 retail curriculum RL

Uses actual Tau2 customer interaction, retail tools, task policy and evaluated terminal checks, without a prescribed action sequence. Initializer: post20 checkpoint-7, SHA256 `210d9192cfad7e5fb999d70fc2e356a8af88788e10d21865fb7d9a95dc43a983`. The stopped medium prototype is preserved but is not the initializer.

Configuration: `configs/rl-retail-tau2-curriculum.json`. Thirty groups × sixteen Transformers batched rollouts; temperature0.7, top-p1;10actor actions and2560generated tokens maximum per episode,8Kactor context. Frozen Qwen3.5-4B runs beside the student as customer, summarizer and NL judge. Structured JSON responses retain their raw content and remove a complete valid Markdown JSON fence for parsing. The teacher endpoint is loopback-only.

`curriculum.json` ranks only the original reserved RL task IDs using reference action count, write actions, assertion count and distinct orders. This is a static complexity proxy, not measured difficulty. Three stages use ten groups each; later stages replay earlier tasks for20% of groups. Tasks30and54 are deferred because their references already require13and12tool calls under the current10actor-action cap. No held-out validation/test task enters training.

Reward is the sum of actual evaluated Tau2 DB, ACTION, ENV_ASSERTION, NL_ASSERTION and COMMUNICATE checks, plus a small unique successful grounded-call bonus, minus schema/grounding/confirmation violations and consecutive-call failure penalties. Empty categories do not contribute artificial neutral credit. Grounding and confirmation guards are conservative text-based heuristics, not semantic correctness oracles. Native Tau2 success is logged separately from shaped reward. Frozen Qwen is an experimental judge, not the official leaderboard judge.

A sixteen-rollout native smoke blocks training on infrastructure failures. Validation uses held-out69and34beforetraining, every10groups, and afterward. Only newly sampled assistant tokens receive policy loss; tool/customer messages and frozen summaries do not. Latest two complete checkpoints, rollout JSON/TXT, checks and W&B advantages are persisted to the private bucket. No periodic Hub dataset/model commits. Runtime failure markers are distinct from successful training completion; deterministic failures do not trigger provider retries.

L40S,180minute timeout,$5.44 maximum tracked reservation. Live viewer uses native success for green and verified partial progress for yellow.
