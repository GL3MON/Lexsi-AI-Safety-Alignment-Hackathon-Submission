# Medium retail workflow GRPO

Config: `configs/rl-retail-medium.json`. Fifty groups, sixteen batched Transformers rollouts per group, 60% medium /20% original clarification /20% other-retention. Post20 checkpoint-7 initializes policy and frozen reference. Fresh AdamW, learning rate 5e-6, seed244, temperature0.7, top-p1.

Medium cases start with two messages and require4–7 generated assistant actions; no hidden reference sequence is rendered. Up to8 actions and1536 generated tokens fit the8K context. Verified subgoal progress avoids tying all partially correct workflows. Full success requires final database fingerprint match.

All reference paths were executed against actual retail tools; medium loader, partition, curriculum and masking tests passed. A three-group16-rollout smoke gates training on reward variance. Greedy114-case validation runs before, every30groups, and after. Two latest complete checkpoints are retained. Bucket-only artifacts, no periodic Hub dataset/model commits. Three-hour cap reserves$5.44.
