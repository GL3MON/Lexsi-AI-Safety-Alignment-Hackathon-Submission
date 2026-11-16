# Tiny Aya GPU diagnostics — 2026-11-09

These runs used the real `CohereLabs/tiny-aya-global` model, BF16 weights, rank-16
LoRA on attention and MLP projections (30,228,480 trainable parameters), SDPA,
gradient checkpointing, and online W&B logging. The H100 MIG slice has 19.625 GiB
total, but competing usage left about 8.905 GiB free before loading the model.
Results are specific to those settings and available memory.

## Smoke and overfit

- Three-sample GPU smoke: three optimizer steps, nonzero gradients, 504 changed
  adapter tensors, checkpoint/final adapter saved, held-out sample evaluated.
  Peak allocated/reserved GPU memory: 7.219 / 7.641 GiB.
- Overfit: the same three samples, 60 optimizer steps, constant learning rate
  2e-4, dropout 0, batch 1. Evaluation intentionally used those training samples.
  Final teacher-forced loss: 0.0001665; token accuracy: 100%.
- All three greedy generated outputs reproduced the reference tool names and
  arguments, including the example with multiple tool calls. These are
  memorization diagnostics, not evidence of held-out tool-use quality.

Online runs:
[smoke](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/zp1dyjbr),
[overfit](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/pwhcb1lk).

Local output root: `outputs/sft-diagnostics/`. `report.json` collects metrics,
run URLs, generation checks, batch-probe results, and the recommended settings.
Generated overfit text is in
`overfit/run-96a9141650/generated_samples.txt`; the adapter is in that run's
`final_adapter/` directory. It is trained only to memorize the three samples.

## Loss memory fix

The initial smoke failed during evaluation. TRL 1.7.1's default chunked loss
materializes an FP32 copy of the 262,144 × 2,048 output-head matrix, a 2 GiB
temporary. The local integration now projects with the native head dtype and
computes cross entropy/entropy in FP32, using 32-token chunks. It changes no
installed package files and checks the pinned TRL version. A regression test
compares loss, gradients, accuracy, and entropy against a direct FP32 reference;
the real-model runs verify updates and exact structured-call memorization.

## Batch limits

Each candidate ran in its own process and attempted two complete optimizer steps
on repetitions of the selected longest conversation. These measure the training
micro-batch size, not the effective batch after gradient accumulation.

| Workload | CPU activation offload | Largest passed | Next batch failed | Peak allocated/reserved at passed batch |
| --- | --- | ---: | ---: | --- |
| Longest of three smoke samples: 1,148 tokens | No | 3 | 4 | 8.162 / 8.514 GiB |
| Longest SFT conversation: 4,563 tokens | No | None | 1 | No successful run |
| Longest SFT conversation: 4,563 tokens | Yes | 1 | 2 | 8.245 / 8.613 GiB |

Allocation failures on this host surfaced as an NVML assertion in PyTorch's CUDA
caching allocator, rather than a clean `OutOfMemoryError`; probe files preserve
the actual `allocator_error` status. These are failed configurations, not counted
as successful or silently retried in the same CUDA process.

For the current full dataset, `configs/sft.json` now enables activation offloading
and, after the diagnostics, was changed by user request to batch size 2 with
gradient accumulation 4 (effective batch 8). Batch 2 did not pass the longest-example
probe under the measured memory budget; the measured working long-example batch
remains 1. At
1,148 tokens, batch 3 is the measured limit without offloading; batch 2 leaves
more headroom. The memory budget is shared and can change. This probe does not
claim a universal maximum for different lengths, quantization or LoRA settings.

Online probes:
[short samples](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/ymsy3bo1),
[long without offload](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/101kn44w),
[long with offload](https://forge.coreweave.com/wandb/harisshragavarchives/tiny-aya-function-calling/runs/2d9xf39e).

No full-dataset training run was started during these diagnostics.
