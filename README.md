# Tau2 SFT and RL with AlignTune

Configurable assistant-only SFT and turn-aware GRPO for Tau2 retail, with a basic telecom solo bridge. Training uses pinned AlignTune trainer extensions, TRL/PyTorch, and W&B online logging.

## Install

```bash
scripts/setup.sh
```

The runtime pins are in `requirements-gpu.txt`; the tested environment is recorded in `requirements-local.lock`. Keep the Tau2 checkout at `../tau2-bench` or set `TAU2_CHECKOUT`.

## SFT

```bash
scripts/run_sft.sh --config configs/sft-retail.json --backend aligntune
```

Configuration supports model/revision, existing adapters, train/validation files, task reservations, context length, LoRA, batch size, accumulation, learning rate, epochs, W&B, and curriculum order. Use `--set KEY=JSON` to override JSON settings. Use `--prepare-only` to verify data without loading weights.

```bash
scripts/run_sft.sh --config configs/sft-retail.json --prepare-only
scripts/run_sft.sh --config configs/sft-retail.json --set batch_size=1 --set gradient_accumulation_steps=8
```

Assistant-only labels are preserved exactly, including selective continuation targets and masked context. `train_order: "tool_call_count"` sorts examples by increasing assistant tool calls and uses sequential sampling each epoch. `--backend trl` remains available for parity diagnostics. Historical `--backend agenttune` resolves to AlignTune without the old package.

## Retail RL

```bash
scripts/run_retail_rl.sh check
scripts/run_retail_rl.sh collect --task-id 59 --set group_size=1
scripts/run_retail_rl.sh train --config configs/rl-retail.json
```

`configs/rl-retail.json` starts from the selected long-SFT checkpoint 75. It uses the frozen Qwen customer and summarizer endpoint at `http://127.0.0.1:8004/v1`. Choose the actor GPU through `RETAIL_RL_GPU`; launch scripts default to MIG0.

Actual Tau2 terminal rewards score whole episodes. Summaries are context transformations, not separately rewarded episodes. The trainer replays each sampled action in its exact context; user/tool observations and memory text are not policy targets. All-zero/tied groups skip updates. Training requires positive temperature and `top_p=1`; evaluation may use nucleus or greedy sampling.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/validate_sft.py --backend both
.venv/bin/python -m tau2_grpo.cli smoke --config configs/small.json
```

The tiny-model SFT check compares AlignTune and direct TRL, with W&B online by default. `configs/sft-aligntune-smoke.json` is a disposable three-sample real-model wiring test. `scripts/smoke_aligntune_rl_optimizer.py` uses explicitly synthetic contrast rewards to verify a real optimizer update; it does not measure Tau2 success.

## Integration details

[AlignTune migration and verification](docs/aligntune-migration.md) describes the extension boundaries. Stock AlignTune SFT would discard our precomputed labels, and stock flat GRPO does not preserve our summarized multi-turn replay, so these are explicit AlignTune `UnifiedTrainerBase` extensions. The legacy AgentTune package and closed-loop deployment service are removed.

Previous datasets, checkpoints, conversations, and evaluation artifacts remain in `outputs/`. The local JSONL viewer is documented in [dataset viewer](docs/dataset-viewer.md).
