# Focused clarification supervision experiment

Start from original Salesforce/APIGen checkpoint50. Use exactly the same 200 training and 24 validation conversations, ordering and split as targeted_clarification_sft200. Only loss_message_indices change. Missing-information traces supervise the last specific missing-field question and subsequent successful identity lookup. Complete-information traces supervise only the immediate identity lookup. Prior failed-email lookups, broad preliminary questions, later account/order tool calls and final prose are context only, masked from loss.

One epoch, batch8, gradient accumulation1, 25 steps, LR2e-5, original system/template. Evaluation/save every5 steps; at most2 checkpoints. W&B online. Choose checkpoint by validation loss.

The original24 development cases remain unchanged. Fresh36 cases have new identities and wording, 18 clarification/18 tool controls, and stricter request-field/unsupported-result heuristics. Compare original50, the earlier128 pilot, and this focused200 pilot with the same fresh grader. Fresh scores never select checkpoints; output review is still required.

prepare.py reproduces labels from the previous200 split. audit.py validates all selected and masked assistant spans and all nonassistant masking. audit log: mask-audit.log. bootstrap.py runs training and both evaluation suites on HF. launch.py records job/cost reservation in the shared budget ledger. All datasets and results are private; secrets are passed only through the HF Jobs secrets field.
