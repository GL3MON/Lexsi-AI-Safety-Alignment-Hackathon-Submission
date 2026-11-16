# Assisted native Tau2 RL

Config: `configs/rl-retail-tau2-assisted.json`. Original reserved RL tasks, real customer interaction, actual retail tools and original task checks. Starts from pre-experiment post20 checkpoint-7 with no positive grounded-call bonus. The earlier unassisted run is stopped; its weights are not the initializer.

Assistance levels: `finish` executes read-only authentication, profile and relevant order/product retrieval; `find` supplies observed authentication only; `none` starts from the original customer opening. The helper obtains identity through the live customer, executes all supplied calls through Tau2, and checks that both database hashes remain unchanged. It never approves or executes a requested mutation or reveals the reference mutation call. Selection of relevant reads is privileged training assistance, not unassisted benchmark evaluation.

Only newly sampled suffix actions enter policy loss. Authentication and order ownership are primed from the supplied observations without awarding credit. Reference-action checks satisfied only by supplied calls are excluded from the additive training reward. Native assisted outcomes remain recorded separately and must not be reported as unassisted success.

A stage advances only when the latest four non-replay groups at that assistance level have at least50% native task success across their sampled rollouts. Later stages replay the preceding easier assistance level in20% of groups. This is a training-success gate, not held-out checkpoint selection. Existing train-only task ordering remains a complexity proxy; it does not force the assistance level to advance.

Thirty groups ×16rollouts; temperature0.7/top-p1;10sampled actions and2560sampled tokens per episode,8Kcontext. Supplied-prefix actions have their own12-action setup cap and are excluded from sampled action counts. Evaluation before, every10groups and after is explicitly UNASSISTED on held-out69and34. Smoke blocks training on setup/infrastructure errors.

Real Tau2 task85 reference integration passed: helper left the DB unchanged; three scripted suffix actions confirmed, modified and reported the jacket change; native DB check passed. This is a pipeline test, not model performance. Thirty-seven checks passed before launch.

L40S,180-minute cap,$5.44reservation. JSON/TXT, assistance levels, supplied action counts, per-rollout advantages and checkpoints are persisted. The live viewer preserves highest-advantage highlights and distinguishes assisted context from sampled actions.

Prefix call records include structured `tool_calls` metadata as well as their rendered text. Citation normalization supports legacy text-only calls only when a single valid call matches its adjacent named tool result. Saved failing summary attempts replay through the unchanged factual validator; malformed and mismatched tool pairs remain rejected.
