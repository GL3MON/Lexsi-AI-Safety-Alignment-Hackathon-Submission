# Retail SDG v2

Run: `.venv/bin/python -u scripts/run_retail_sdg_v2.py`. Output: `outputs/retail-data/retail-sdg-v2-700`.

Quotas: 200 recovery conversations (40 of each corruption type), 400 ordinary clean conversations and 100 calculator-focused clean conversations. Total 700, split 630 train / 70 validation by scenario family. Twelve independent workers share the two Qwen3.5-4B FP8 endpoints, each serving four sequences concurrently.

Calculator scenarios include prospective combined refunds, same-size/material variant price differences, and budget balances or overages. The teacher must retrieve relevant order/catalog prices. Expressions must perform arithmetic using only observed monetary values, previous calculator results, or a customer-provided budget. The expected answer must actually be returned by calculate and appear correctly in the final answer. Budget wording is checked for within/over direction.

During synthetic construction, the teacher's response schema requires a calculate call once all arithmetic inputs are available and the requested numerical result has not been obtained. The teacher still derives the expression; no expected value is supplied in its prompt. This construction constraint is not a student inference rule. Audit records explicitly mark it; training JSONL contains only the conversation and loss metadata.

Every turn has deterministic role checks and failed drafts stay out of the saved conversation. Every accepted conversation additionally passes focused Qwen role review, semantic review, real-tool and goal-state checks, and the 8192-token/masking checks. The model-based reviews are quality screens, not formal guarantees. Existing v2 accepted samples are retained when the scenario mix expands.
