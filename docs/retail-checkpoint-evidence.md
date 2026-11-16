# Evidence for the checkpoint-level write-up

This local companion maps the behavioral claims in [the Hugging Face draft](huggingface-retail-agent-writeup.md) to saved outputs. No new inference or training was launched for this review. Checkpoint numbers are specific to each training stage. Historical runs are development diagnostics, not a single controlled benchmark.

| Artifact | Evidence | What it supports |
|---|---|---|
| Raw Aya, native EOS | [Saved artifact](../outputs/retail-eval/raw-aya-base-task38-native-eos/episode.conversation.txt) | Invented payment/order identifiers and generated tool-result text contradicted by the environment. |
| Hermes comparison | [Saved artifact](../outputs/diagnostics/hermes-base-clarification-comparison/conversations.txt) | Missing-field handling, invented identity, identifier transformation, and failure to advance. |
| Initial retail SFT | [Saved artifact](../outputs/retail-rl/base-vs-retail-sft-46-v2/comparison-checks.txt) | Task 59: partial checks passed, final state and requested mutations failed. |
| Initial retail SFT transcript | [Saved artifact](../outputs/retail-rl/base-vs-retail-sft-46-v2/retail_sft.conversation.txt) | Repeated reads, unsupported quantity claims, and repeated non-call completion strings. |
| Short / copy-repair / long SFT | [Saved artifact](../outputs/diagnostics/completed-sft-stages-four-scenarios/conversations.txt) | Matched scenario text shows email/name mutations, selective fixes, and guessed ZIP/user ID. |
| Three-task transfer comparison | [Saved artifact](../outputs/retail-eval/three-test-tasks-three-sft-stages/scores.csv) | Nine local attempts across three stages; all reported zero task reward. |
| Transfer comparison settings | [Saved artifact](../outputs/retail-eval/three-test-tasks-three-sft-stages/config.json) | Exact checkpoints, task IDs, generation settings, and local judge scope. |
| Long-SFT task 38 | [Saved artifact](../outputs/retail-eval/three-test-tasks-three-sft-stages/long-500/task-38.conversation.txt) | Repeated email mutation and failed lookup. |
| Glaive SFT vs RL | [Saved artifact](../outputs/diagnostics/glaive-rl-checkpoint16-comparison/summary.json) | Identical saved category-level probe counts; not evidence of improvement. |
| Glaive seen-case clarification | [Saved artifact](../outputs/diagnostics/glaive-rl16-failure-case-comparison/conversations.txt) | Both evaluated checkpoints called with empty required text. |
| Hint-distilled step 200, task 69 | [Saved artifact](../outputs/retail-hint-training/run-20261112T082253/evaluation/step-000200/task-69-seed-100042.conversation.txt) | Six-turn successful cancellation with tool-confirmed final state. |
| Horizon audit | [Saved artifact](../outputs/diagnostics/retail-horizon/report.md) | Malformed attempt rates by turn depth; different task samples, no causal comparison. |
| Older replay-SFT role failure | [Saved artifact](../outputs/retail-eval/replay-sft-checkpoint50-task34/episode.conversation.txt) | Assistant speaks as customer and invents identity facts. |
| Older data role contamination | [Saved artifact](../outputs/diagnostics/retail-replay-sft-audit/customer-perspective-candidates.json) | Customer-perspective text present in supervised assistant labels. |
| Adapter load audit | [Saved artifact](../outputs/diagnostics/retail-replay-sft-audit/adapter-load-audit.json) | Evaluated checkpoint loaded correctly; all 504 adapter tensors changed. |
| Sampler exposure audit | [Saved artifact](../outputs/diagnostics/retail-replay-sft-audit/sampler-exposure-check.json) | Known contaminated training record encountered before checkpoint 50. |
| Matched pre-SFT first turn | [Saved artifact](../outputs/retail-eval/pre-retail-sft-glaive-rl17-task34/paired-first-turn.json) | Pre-SFT policy also invented identity under identical prompt IDs. |
| Current 700-sample audit | [Saved artifact](../outputs/diagnostics/retail-sdg-v2-final-audit/REPORT.md) | Replay and masking passed; 117 conversations flagged for prose grounding. Report predates the decision to retain all 700 and launch. |
| Current SFT configuration | [Saved artifact](../configs/sft-retail-sdg-v2-all700.json) | Post-Glaive-RL17 initialization and current training settings. |
| Additive reward implementation | [Saved artifact](../src/tau2_grpo/rl_rewards.py) | Actual criteria, grounding, unique-call credit and repeated-failure terms. |
