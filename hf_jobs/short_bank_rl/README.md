# Scaled short retail GRPO

Continues the short pilot's checkpoint 8 (adapter SHA256 `40721ea0ea8664a6c0fb70142eb8a31ef8e57aabb97fb86423718f89df8f370f`) using the completed 3,000-case short bank. Training pool: 2,400; validation: 300; reserved test: 300. Whole customer worlds remain disjoint. Hidden verifier routes/state are not rendered to the actor.

Balanced sampling covers each of 30 families once per 30 groups, with a fresh randomly chosen world per family. Target: 60 groups × 8 sampled rollouts = 480 episodes, not a full pass over all 2,400 training cases. Maximum four actions, 192 tokens per response, 8K context. Transformers BF16 generation, temperature 0.7/top-p 1; online GRPO through AlignTune, frozen initial reference, LR 5e-6, KL .02, clip .2. Reward +1 for completing the short goal, -1 otherwise. Native Tau2 rewards are not evaluated. Infrastructure errors remain null and block a group update; tied groups are skipped.

Greedy validation uses one fixed held-out world case per family (30), before training, each 30 groups, and after training. The remaining validation cases and all test cases are not used for updates. This is a development subset, not a full benchmark. W&B logs per-group results, per-rollout advantages and readable sample tables. Checkpoints are saved after successful updates with only the last two retained.

Launch: `.venv/bin/python hf_jobs/short_bank_rl/launch.py --submit` (paid; don't rerun if a job is already active). Read-only watcher: `.venv/bin/python -u hf_jobs/short_bank_rl/watch.py`. Current job metadata: `job.json`. Local monitoring: `outputs/hf-jobs/<job_id>/job.log`, with readable conversations downloaded under `bucket/`.

A private bucket input archive supplies the scenario bank; its SHA256 and the pilot adapter hash are verified remotely. There are no periodic Hub dataset/model output commits. The mounted private bucket stores complete checkpoints and rollout files. L40S, 90-minute hard timeout, maximum reservation $2.74; original cumulative budget ceiling $20 remains enforced by the launch ledger. Timeout can end the job before its target groups; completed checkpoints survive in the bucket.

Validation: 54 tests passed, including actual collector execution with canonical reference actions for all 30 families, split/sampler checks, and native/short-pilot regression tests. Real CLI check confirms the checkpoint, system/template contract, 2,400 training cases and output reserve.
