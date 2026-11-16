# Setup and operation

## Working directories

The local layout is:

```text
/home/harishr/
├── tau2-bench/                  # editable benchmark checkout and domain data
└── tau2-agenttune-grpo/         # this project
    ├── .venv/                  # independent Python environment
    ├── configs/
    ├── docs/
    ├── outputs/
    ├── scripts/
    ├── src/tau2_grpo/
    └── tests/
```

The package is installed editable. Source changes are visible without reinstalling;
changes to packaging or dependencies require installation again. The original
notebook is provenance only and is not loaded at runtime.

## Requirements and installation

Use Linux with `bash`, Git, `uv`, Python 3.12, network access for package/model
downloads and a CUDA-capable NVIDIA GPU for training. The project metadata accepts
Python >=3.12,<3.14; the setup script specifically requests 3.12. A CPU-only machine
can execute configuration checks and a Gym smoke test with the needed tau2 Gym
dependencies installed. The provided setup installs the full GPU stack.

```bash
cd /home/harishr/tau2-agenttune-grpo
make setup
```

The script reuses `.venv` when its Python executable exists, otherwise creates it.
It installs this project and the sibling tau2 checkout with its `gym` extra, the
requirements in `requirements-gpu.txt`, and exact CUDA 12.6 torch/torchvision builds.
It then runs `doctor`. It does not replace an existing environment automatically
or verify that an existing environment uses the requested Python version.

To install against a different tau2 location:

```bash
TAU2_CHECKOUT=/absolute/path/to/tau2-bench bash scripts/setup.sh
.venv/bin/tau2-grpo smoke --tau2 /absolute/path/to/tau2-bench
```

`TAU2_CHECKOUT` is read by the setup script; CLI runs use the `--tau2` argument.
Point both at the same installed checkout. `--tau2` validates the directory and
records its Git revision; it does not change which installed `tau2` module Python
imports. Domain data comes from the installed editable tau2 checkout.

The setup sets `UV_FROZEN=1` and uses `uv pip`, leaving tau2's `uv.lock` untouched.
It uses the PyTorch CUDA 12.6 index with `unsafe-best-match`, allowing resolution
across that index and the default package index. The torch and torchvision CUDA
builds are explicitly pinned. The project `pyproject.toml` deliberately has no
training dependencies: `pip install -e .` alone does not install the training stack.

## Dependency versions and reproduction

The direct contract pins torch 2.11.0, transformers 5.17.0, TRL 1.7.1, PEFT 0.19.0
and jmespath 1.0.1. AgentTune comes from Git commit
`d4db4c12d52342227c4113a70183d0f64aa2f345`. Training checks the four core package
versions and AgentTune's installed Git origin, then checks TRL's environment API.
Accelerate and datasets have version ranges in the requirements file.

`requirements-local.lock` is a freeze of the installed local environment. It
contains exact package versions and editable absolute paths for this machine;
it is not a portable cross-platform lockfile. To restore it after creating a venv:

```bash
uv pip sync --python .venv/bin/python requirements-local.lock   --index https://download.pytorch.org/whl/cu126   --index-strategy unsafe-best-match
```

Adjust editable paths deliberately when moving machines. Normal `make setup`
uses the direct requirements rather than this snapshot, so transitive dependencies
may resolve differently. Preserve the snapshot and actual run artifacts when
comparing experiments. Model `main` is resolved to a commit when available and
recorded in `runtime.json`; set an exact `model_revision` to repeat it later.

## Command reference

Run Make targets from the project root. Activation is optional because commands
use `.venv/bin` explicitly.

| Command | Behavior |
| --- | --- |
| `make setup` | Install dependencies and run doctor |
| `make doctor` | Report required package metadata and CUDA availability |
| `make smoke` | Run one scripted, evaluated telecom Gym episode without loading a model |
| `make small` | Start the full workflow using the 0.6B, one-step preset |
| `make train` | Start the full workflow using the 1.7B, ten-step preset |
| `make test` | Run dependency-free configuration and syntax unit tests |

```bash
.venv/bin/tau2-grpo --help
.venv/bin/tau2-grpo config --config configs/local.json
.venv/bin/tau2-grpo doctor
.venv/bin/tau2-grpo smoke --output ./outputs
.venv/bin/tau2-grpo train --config configs/small.json --output ./outputs
```

All commands load and validate the selected config. `config` prints effective
settings without checking tau2 or importing the training stack. Other commands
validate the tau2 path. `doctor` exits with status 1 for missing listed packages or
unavailable CUDA; it does not exercise full AgentTune imports, run a CUDA kernel,
verify exact package pins, check free memory or train a model. Training performs
additional compatibility checks. `smoke` checks Gym and writes a new run directory,
then exits before importing the GPU training stack or loading a tokenizer.

## GPU selection and memory

The recorded machine has an H100 split into 20 GB MIG slices. The validation record
observed only about 8.88 GiB free on the selected slice. These are historical
measurements; inspect current availability before a run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 .venv/bin/tau2-grpo doctor
CUDA_VISIBLE_DEVICES=0 .venv/bin/tau2-grpo train --config configs/small.json
```

If your host exposes MIG UUIDs, supply the intended available MIG identifier as
`CUDA_VISIBLE_DEVICES`. The workflow defaults to `0` only when the variable is
absent. It requires one visible device and `WORLD_SIZE=1`. There is no distributed
launcher support. A small preset can still exceed available memory; policy length,
tool results, rollout budget, reference-model behavior and optimizer state all
matter. The baseline is released before the background trainer loads its model.

## Logs, stopping and recovery

The CLI prints the new `outputs/run-<id>` directory immediately. For a persistent
shell log, create its parent first and capture both output streams:

```bash
mkdir -p outputs/logs
set -o pipefail
.venv/bin/tau2-grpo train --config configs/small.json 2>&1 | tee outputs/logs/small.log
```

The main thread prints background progress approximately every 30 seconds. The
training/gate wait budget is four hours, set by `MAX_WAIT_SECONDS` in the source;
it is not configurable via JSON and is not a runtime estimate.

Use Ctrl-C to stop an interactive CLI run and ensure the Python process has exited
before starting another. The background worker has no cooperative cancellation
or resume API here. In an embedding or notebook, interrupting only the waiting
thread can leave work running; terminate that process before retrying. Partial
run files remain for inspection. Re-running always creates a new run ID; saved
trainer state is not automatically resumed.

No serving process is started. An approved gate updates only that run's local
adapter selection manifest. See [artifacts](artifacts.md) for interpreting it.
