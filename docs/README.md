# Documentation guide

Start at the [project README](../README.md) for commands and status.

| Document | What it covers |
| --- | --- |
| [Setup and operation](setup.md) | Environment installation, CLI, GPU selection, logs, stopping runs, reproduction |
| [Configuration](configuration.md) | Every JSON option, defaults, presets, derived trainer settings, environment variables |
| [Architecture and training](architecture.md) | Module boundaries, rollout API, GRPO, splits, AgentTune loop, gate and evaluation |
| [Artifacts and results](artifacts.md) | Run files, schemas, phase names, inspection commands and result interpretation |
| [Troubleshooting](troubleshooting.md) | Installation, CUDA, memory, sparse rewards, errors and partial runs |
| [Development](development.md) | Source map, tests, extension points and current limitations |
| [Local validation record](../VALIDATION.md) | Checks actually performed and work not yet verified |
| [Historical notebook reference](reference.md) | Original rationale and upstream links; notebook-specific paths are historical |

The executable source is the authority for behavior. These guides describe this
standalone local project, including its current limitations. They do not establish
that full GPU training has completed.
