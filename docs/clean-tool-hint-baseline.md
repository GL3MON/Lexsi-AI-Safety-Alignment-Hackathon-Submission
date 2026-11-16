# Clean prompt and manual-hint baseline

The audit found a synthetic-data construction bug: the old generator replaced
a documentation placeholder instead of the real catalog. All 405 tool-result-copy
fixtures therefore contained two JSON catalogs, one stale and one expanded.
The other legacy prompts also contain literal documentation delimiters. The
exported tokenizer template includes an unmatched opening tools tag in explanatory
prose; it is not a second catalog, but makes delimiter counting ambiguous.

`glaive_prompts.system_prompt` now builds the entire system prompt from the
actual row schemas, with exactly one catalog. `clean_tokenizer_template` removes
the unmatched documentation tag from the in-memory tokenizer template without
changing control tokens or historical checkpoint files. The source Jinja template
is fixed too. The Glaive RL runner applies this cleanup for future runs.

The original SFT data and historical model artifacts are retained. The corrected
RL data is `outputs/glaive-rl-signals-v2/train.jsonl`; `configs/rl-glaive.json`
now points to it. The old checkpoint was trained with the old fixtures; fixing
data does not retroactively change its weights. The duplicate-catalog issue
alone does not establish that SFT must be repeated.

Unicode audit: the original French target contains U+00E8. JSON spelling with
an escaped Unicode code point parses to the same value; a literal backslash-u
string is different. No literal Unicode escape values were found in the gold
arguments. U+00E3 was a real copying mismatch. The new 20-case baseline uses
ASCII values only and does not test translation or multilingual text.

The 20 fixed cases comprise eight missing-information, six copying, and six
correction cases. Their schemas and reference calls are validated. The exact
rendered prompt is exported per case and arm. Each case is tested with and without
a manually written next-response instruction, with seeds 42/43, temperature 0.7,
and top-p 1.0, on frozen RL checkpoint-16. No teacher model generates these hints,
no backend API executes, and no optimizer update runs. This is a controlled
manual-hint-following baseline, not HINT distillation or benchmark performance.
Clarification scores are heuristic: a targeted request with no call passes;
inspect wording for unnecessary additional requests or other quality problems.

Artifacts: `outputs/diagnostics/clean-manual-hint-baseline/` contains cases,
manifest hash, audit report, rendered prompts, actual template, results, and TXT
conversations. Historical failed preliminary evaluations are not mixed into the
final results; a run is complete only when `run.json` records 80 completions.

```bash
.venv/bin/python -m scripts.prepare_glaive_rl_signals
.venv/bin/python -m scripts.build_clean_tool_cases
.venv/bin/python -m scripts.audit_clean_tool_data
CUDA_VISIBLE_DEVICES=MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04 \
  .venv/bin/python -m scripts.evaluate_clean_manual_hints
```

## Completed result

Twenty cases, two seeds, eighty total completions, frozen RL checkpoint-16:

| Category | No hint | Manual hint |
|---|---:|---:|
| Missing information | 3/16 | 16/16 |
| Copying | 10/12 | 11/12 |
| Corrections | 12/12 | 12/12 |
| Total | 25/40 | 39/40 |

All 16 hinted clarification outputs were inspected for actual targeted requests.
Hints were appended to the latest user message as direct next-response guidance,
without the paper evaluation wording. This demonstrates hint-following in that
setup; it does not establish HINT distillation efficacy or equivalence between
user-message hints and privileged teacher contexts. No weights changed.
