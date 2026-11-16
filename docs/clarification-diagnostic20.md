# Small SFT ask-versus-call diagnostic

Question: can SFT19 learn to change its next action when one required input becomes available?

Train on20 matched pairs (40 examples): five pairs each for email, first name, last name and ZIP. Each absent/present pair has the same retail system prompt and customer request; the supplied version only appends the missing value. Missing cases target a specific clarification. Supplied cases target the real lookup with exact arguments. Separate20 matched pairs (40 examples) use fresh identities and states, with the same wording structure. They test identity transfer, not broad dialogue generalization.

1. Greedy free generation on all40 train and40 held-out examples using SFT19.
2. Deliberate memorization test:50 optimizer steps, batch4/accumulation1, LR5e-5, equal per-example mean NLL for ask/call, no retention mixture. System/user labels are masked; one assistant decision is supervised. The final step is evaluated; there is no checkpoint selection based on held-out success.
3. Repeat the exact same80 evaluations with the final adapter.

Report ask accuracy, grounded call accuracy, decision-type accuracy and the number of matched pairs where both halves pass. Save every generated answer for review. Correct calls execute against isolated synthetic customer records. This is a next-decision rubric; it does not evaluate complete retail workflows or certify every free-text claim.

Interpretation:

- Strong performance on seen and fresh identities supports learning of the controlled decision pattern.
- Strong seen-case performance with weak held-out performance suggests memorization or identity-transfer difficulty.
- Weak seen-case performance calls for checking labels, template, adapter updates, optimization strength and training duration before scaling.

The dataset uses fresh worlds200–219 and300–319, outside previous generated banks. Each world supplies one matched pair. Original task structure is seeded from the prior22-task RL pool, so it is not an independent native Tau2 benchmark. Longest training sequence:4453 tokens; no truncation. Gold calls execute successfully, target masks are audited, and pair splits are disjoint.

Files: `outputs/retail-data/clarification-diagnostic20/{train,heldout}.jsonl`, `manifest.json`, `examples/*.txt`. Config: `configs/sft-clarification-diagnostic20.json`. Evaluator: `src/tau2_grpo/sft_diagnostic.py`. This experiment is SFT-only and should not be promoted as a production checkpoint.

Runs on the funded HF account because local MIG0 had only1.66GiB free. Maximum30 minutes/$0.94, with online W&B and private bucket checkpoint/log persistence. No automatic further training. Watcher: `hf_jobs/clarification_diagnostic/watch.py`; launch metadata in its adjacent `job.json`.

Submitted job: https://huggingface.co/jobs/notGL3MON/6af9cfe623985fb9cd837af7

## Evaluation correction

The first generation evaluator accidentally reused a stateful context controller across independent cases. An audit found one unique rendered prompt across all80 cases; those generation scores are invalid and are hidden in the viewer. The trained adapter,50 optimizer steps and teacher-forced losses are unaffected. Corrected evaluation constructs one controller per independent case, checks its rendered messages against that case, and rejects duplicate prompt IDs. Regression tests cover multiple distinct cases. Earlier90-case weighted-SFT/DPO evaluations used90 unique prompts and their episode factory already creates separate controllers.

Evaluation-only correction: https://huggingface.co/jobs/notGL3MON/6af9d56223985fb9cd837e64 . No retraining.

## Corrected results

Independent audit found80 distinct prompts per evaluated model, with exact raw/rendered case identity and no assistant targets in the inputs. Both models evaluated all40 training and40 held-out cases. SFT19 scored21/40 on each split (questions1/20, calls20/20). After50 SFT updates, both splits scored40/40 (questions20/20, calls20/20), and20/20 matched pairs passed both halves. The learned rule transfers to fresh identities under the same wording structure; broader phrasing/workflow retention remains unmeasured. Corrected artifact: `outputs/hf-jobs/6af9d56223985fb9cd837e64/RESULTS.md`.
