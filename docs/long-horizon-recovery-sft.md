# Long-horizon retail recovery curriculum

Generate a 12-conversation review pilot:

```bash
.venv/bin/python scripts/generate_long_horizon_recovery.py --count 12 \
  --output outputs/retail-data/long-horizon-recovery-pilot \
  --endpoint http://127.0.0.1:8004/v1 --model qwen3.5-4b
```

Qwen on MIG2 writes recovery explanations and dialogue wording. A deterministic
expert plan serializes and executes the calls against isolated synthetic Tau2
DBs. This is guided synthetic SFT data, not free-running Qwen performance or a
Tau2 benchmark reward. Synthetic account, order, product and variant IDs are
new; products and tool semantics are copied from the retail domain.

Each conversation resolves six orders: two cancellations, two returns, one
shipping-address change, and one conditional boot exchange. Every mutation is
preceded by explicit confirmation. Exchanges enforce availability, size,
material, and a lower actual price. A third of families keep the boots because
no qualifying cheaper variant exists. Claims distinguish actual cancellation
refunds from pending return/exchange processes.

Eight intended failures (seven on the no-exchange branch) exercise lookup
clarification, malformed read and write JSON, invalid argument names, delivered
order cancellation, mistyped payment IDs, late malformed JSON, and invalid
calculator expressions. Each failure is parsed/validated or executed through
the real tool implementation, must leave DB state unchanged, and must be
followed by an explained correction. Failed assistant turns are context only
and are excluded from every loss mask.

`records/*.txt` are complete readable conversations, with masked faults labeled.
`records/*.json` include real tool results and final-state checks. `sft.jsonl`
and `validation.jsonl` contain one supervised correct assistant turn per window.
Earlier assistants, fault turns, customer/tool observations, memory and policy
are masked. Targets are covered exactly once. Windows exceeding 8K retain the
original request, a deterministic memory built only from prior visible facts,
and a recent verbatim tail. Explicit target IDs must remain present in context.
The original full traces are never truncated or overwritten.

The pilot has 10 synthetic training families and 2 validation families; every
window from a family stays in the same split. Shared scenario patterns mean
this validation is not a held-out behavioral benchmark. Qwen wording is
constrained by required phrases and invention checks; those checks are not a
proof of every natural-language nuance. Inspect the pilot before scaling.
No training is launched by this generator. Accepted records support resuming
with the same count/configuration; use a fresh directory for a different pilot.
