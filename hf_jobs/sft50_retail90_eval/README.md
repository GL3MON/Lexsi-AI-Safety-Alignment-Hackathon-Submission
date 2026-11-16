# Diagnostic SFT50 on the fixed90-case retail development set

Inference only. Uses the final adapter from the controlled40-example/50-step SFT diagnostic, SHA256 eae5e7bc4d3d2338ac3afaa2ec9a549808a6fa57f17844f3e95d4feea11a6e1d. It does not retrain or use DPO. The controlled diagnostic passed40/40 training and40/40 new-identity cases after its evaluator was corrected and independently audited.

Broader evaluation: same90 short-scenario-bank cases, all30 families in worlds80,85,86. Compare with the saved SFT19 baseline57/90 (clarification5/24, other52/66). Settings retained: Transformers BF16, saved SFT retail system/Jinja, greedy temperature0, top-p1, max192 response tokens, at most4 actions,8K context. Each episode gets its own context controller. Runtime guards reject duplicate initial prompts or rendered/raw case mismatch.

Both case IDs and original baseline IDs must match. Reference routes, valid alternate routes, tool execution and final mutable-state hashes are unchanged. These are synthetic development scenarios, not native Tau2 benchmark results. The90-case test checks whether the narrowly learned clarification pattern transfers to other wording and histories and whether broader tools regress.

The source archive/scenario bank is read-only mounted from the new account's private input bucket; the selected SFT50 and outputs use its retail-experiments bucket. There are no input uploads or Hub model/dataset output commits in this inference job. Frozen-base gating access is verified before paid submission.

Budget:15-minute hard cap, maximum$0.49 at currentL40S rates. Saved results are under the job bucket prefix; current metadata in job.json. Watch with `.venv/bin/python -u hf_jobs/sft50_retail90_eval/watch.py`. Local mirrors: outputs/hf-jobs/<job_id>/. New interactive traces appear in the live viewer on port8766.
