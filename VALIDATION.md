# Local validation — 2026-11-09

- Isolated Python 3.12 environment created in `.venv`; 279 packages installed.
- `uv pip check`: all installed packages compatible.
- PyTorch 2.11.0+cu126, transformers 5.17.0, TRL 1.7.1, PEFT 0.19.0,
  AgentTune 1.1.3 from commit d4db4c12d52342227c4113a70183d0f64aa2f345.
- CUDA is available on NVIDIA H100 PCIe MIG 2g.20gb.
- AgentTune trainer/closed-loop imports and TRL `environment_factory` interface verified.
- Real CUDA allocation and kernel reduction succeeded (sum = 16).
- At verification the selected slice had about 8.88 GiB free of 19.62 GiB;
  availability changes with other workloads.
- `make smoke` passed inside the new environment: actual telecom solo tools,
  clean termination and evaluated task reward were saved under
  `outputs/run-db6b8ef791/`. The zero reward is an expected wiring result,
  not a trained-policy score.
- Three dependency-free unit tests passed; Ruff lint and formatting passed;
  setup shell syntax check passed.

Full GPU training, adapter updates, validation promotion and model reload have
not been executed. No learning gain is claimed. Use `make small` on an available
GPU slice to exercise the full experiment. The default experiment is intended
for a GPU with more free memory than this shared slice currently has.

The source notebook was copied into a standalone Python workflow; subsequent
runs do not read or execute the notebook. The sibling tau2 checkout and its
lockfile were left unchanged.


## Documentation verification

The standalone guides cover setup, all 20 configuration fields, module/workflow
contracts, artifacts/results, troubleshooting and maintenance. Local links, code
fences and shell-example syntax were checked in the README and new guides. CLI
help matched the documented commands and options, and all three existing unit
tests passed again. Documentation work did not start model training or change
runtime source files. Historical notebook references retain their original paths
and are explicitly labeled as historical.
