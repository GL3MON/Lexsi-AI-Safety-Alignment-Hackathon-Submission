# Short retail subgoal GRPO

Switches from long native Tau2 tasks to1-4-turn, read-only subgoals derived from the original22-task RL partition. Starts from focused SFT run2190c37f90/checkpoint25; it does not initialize from the cancelled long-run RL update. Fixed customer replies replace the model customer. No Qwen/vLLM server is needed in this job.

Eight kinds: email lookup, complete name/ZIP lookup, missing ZIP, missing surname, missing first name, correcting a failed email lookup, retrieving a verified user profile, and authentication-to-order-status. Actual Tau2 RetailTools execute correct calls. Incorrect/non-call output at call-required steps fails. Clarifications must request the missing field without known fields, generic extra-information requests, fabricated results or identifiers. Truncated answers fail immediately. Success requires every step of the subgoal, with +1 for success and -1 otherwise; no partial credit for filler. These are curriculum rewards, not native Tau2 task-success scores.

16 groups,8 rollouts/group, max4 agent actions,128 response tokens, temperature0.7/top_p1. Exactly saved SFT system/Jinja template. Standard clipped GRPO/reference KL and per-rollout W&B advantages. Tied groups are skipped honestly. Best practices checks:40 tests plus all22 training task mappings and3 validation mappings pass.

Nine greedy held-out mini-scenarios (three scenario kinds across three reserved validation tasks) run before and after training. They do not select updates or checkpoints. Short episodes remain inside the8K context; unexpected overlength is an infrastructure failure, not silent truncation or invented zero labels.

90-minute maximum, within the original$20 cumulative authorization. Private bucket mounted for checkpoints (last2 complete), groups, readable rollouts and validation outputs. No periodic Hub output uploads/commits; launch source is embedded in job configuration rather than uploaded to a dataset. watch.py is read-only and retrieves logs/metadata/text.
