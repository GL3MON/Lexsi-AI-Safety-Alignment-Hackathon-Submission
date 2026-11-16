# End-to-end retail SFT and automated assisted-RL follow-up

This run reuses existing verified synthetic conversations; it generates no new training dialogue. Initializer is pre-experiment post20 checkpoint-7, not a read-loop RL checkpoint. The lineage includes earlier Salesforce multi-turn SFT; this cleaned synthetic branch is new to it.

Training599records:353clean full conversations,146recovery full conversations,100clarification/retention records. Validation104records:40clean,20recovery,44retention. The original559full dialogues passed real tool replay. All703combined records independently passed exact Jinja-generation-mask and installed TRL-collator checks. Only selected assistant turns, including their stop tokens, receive labels. User/tool messages and intentional failed calls are masked.

System prompts are canonicalized to the exact rollout/RL retail policy; all assistant, customer and tool messages and original loss-message selections are preserved. All input schemas used by successful calls match. The unused source-only get_item_details catalog entry is omitted because it is not exposed by the canonical policy and has no target examples. All records fit8Kwithout truncation.

Coverage includes cancellations, delivered returns, shipping changes, status/price lookup and calculator scenarios. Item modification and exchanges are not demonstrated by this corpus; no direct jacket-change learning claim is made. The follow-up native task curriculum starts with the demonstrated mutations before introducing other original RL tasks.

SFT: AlignTune,2epochs, requested microbatch8 × accumulation2, effective16, LR2e-5, BF16, random ordering, example-normalized NLL with unit weights, MLPchunk512, no activation offloading, online W&B. A discarded two-step optimizer memory probe tests the longest record at batch8; on memory pressure only, fall back4×4or2×8, preserving effective16. Eval/save every10steps, retaining best/latest with limit2. About76optimizer updates are expected at batch8.

After training, load the selected final adapter, run ONE unassisted greedy Tau2 task69 with native checks and frozen Qwen customer/summary/judge, and save JSON plus readable .conversation.txt. Then the SFT container exits. A separate one-shot watcher verifies the completed job, selected model hash and scored rollout before submitting assisted RL from that exact adapter. A scored zero-reward rollout still permits the explicitly requested follow-up; an unscored infrastructure failure blocks it. No RL starts before the test finishes.

SFT+rollout: L40S,90minute cap,$2.74reservation. Assisted RL: separate L40S job,120minute cap,$3.64reservation;30groups ×16rollouts, no standalone grounded-call bonus, success-gated finish→find→unassisted assistance, easier-level replay. The pending RL reservation is consumed once on submission, not charged twice in the ledger. Combined maximum reservations leave approximately$2.55in the tracked account budget.

Queue state: `hf_jobs/assisted_after_e2e/queue-status.json`. Watcher log: `outputs/logs/e2e-followup-queue.log`. Parent status/log: `outputs/logs/retail-e2e-sft-watch.log`. No periodic model/dataset Hub commits. Private bucket artifacts and online W&B provide progress.
