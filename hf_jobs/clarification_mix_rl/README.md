# Continued clarification-focused GRPO, 16 rollouts

Submitted: https://huggingface.co/jobs/GL3MON/6af99e2b23985fb9cd8347cf. The user authorized the run against approximately$3.50 available credit and requested a90-minute hard cap.

Initializer: selected post-RL clarification/retention SFT checkpoint19 (`run-2229ede72f/final_adapter`). Its SHA256 is `98733d46f67d4705145ae00164e757458fe0bca8bacaf97639b4e3d7292e8811`. This continues the SFT output, not the older RL28 adapter. Post-SFT greedy evaluation is18/30 versus17/30 before: partial-ZIP clarification newly passes, all17 previous successes are retained. Other clarification families remain weak.

Target:90 groups ×16 rollouts =1,440 training episodes. Sampling has two clarification groups per retention group (60 /30), rotating across all8 clarification and22 other families. Each group uses one case so relative advantages compare the same task; cases come only from the bank's2,400 training scenarios. Sixteen CPU environment workers request batched inference from one Transformers BF16 policy on an L40S. HF lists8 vCPUs,62GB host RAM and48GB GPU VRAM for this flavor.

Settings: temperature0.7/top-p1, up to4 actor actions,192 response tokens per action,8K context, LR5e-6, clip0.2, reference KL0.02, one update per mixed-reward group. Rewards are+1/-1 for the complete short goal, not native Tau2 task rewards. Tied groups skip updates. Online W&B logs samples and rollout advantages; greedy validation runs before, every30 groups and after. Last2 complete checkpoints are kept. The true scenario family is now logged instead of the unused default `email_lookup` setting.

`configs/rl-retail-clarification-mix.json` contains the exact runnable config. `launch.py` defaults to the original$20 cumulative cap and refuses to submit if it would exceed it. `plan.json` is a read-only submission preview. The submitted90-minute hard cap reserves at most$2.74 (91 billing minutes plus$0.01), leaving$0.76 from the stated$3.50 available. Existing conservatively reconciled spending was$18.85; the equivalent cumulative authorization for this run is$22.35. These are upper-bound estimates, not an invoice or a verified HF account balance. The account must have sufficient available credit; this code does not top up funds.

Submitted command (do not rerun while active):

```bash
.venv/bin/python hf_jobs/clarification_mix_rl/launch.py --timeout-minutes 90 --budget-cap 22.35 --submit
.venv/bin/python -u hf_jobs/clarification_mix_rl/watch.py
```

The job reuses the existing private scenario-bank archive and SFT adapter. Source is embedded in the job environment; there are no Hub output uploads/commits. Checkpoints, episode JSON/text and summaries persist in the private mounted bucket. A hard timeout can still stop an incomplete run; completed checkpoints remain recoverable. The90-minute timeout takes priority over the target90 groups; the number actually finished will be reported from the saved group artifacts.

Preflight passed: exact initializer hash, prompt/template,2,400 allowed training scenarios and output reserve. Forty-seven regression and curriculum tests passed, including the2:1 sampler, all-family reference trajectories through the actual collector, native rewards and probability replay tests.
