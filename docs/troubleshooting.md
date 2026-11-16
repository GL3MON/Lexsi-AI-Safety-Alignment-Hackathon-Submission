# Troubleshooting

Begin with `make doctor`, then `make smoke`, then the small training preset.
Preserve the printed run directory and shell traceback when a command fails.
Check [local validation](../VALIDATION.md) before assuming a training path has
already been exercised.

| Symptom | Explanation and next action |
| --- | --- |
| `.venv/bin/tau2-grpo` missing | Run `make setup`. Installing only this package does not install GPU/Gym dependencies. |
| Setup fails during download/build | Verify network, disk space and Git access, then rerun setup; it reuses the existing venv. Inspect the actual failing package. |
| Existing venv uses wrong Python | Setup reuses it. Create a separate Python 3.12 environment deliberately rather than expecting setup to replace it. |
| `Missing tau2 checkout` / invalid checkout | Supply `--tau2 /absolute/path` and install that same checkout editable with its Gym extra. |
| CLI points at one checkout but imports another | `--tau2` does not override imports. Reinstall the intended editable tau2 checkout and inspect `tau2.__file__`. |
| `CUDA available: False` | Check device visibility, driver access and CUDA-enabled PyTorch. Setup targets CUDA 12.6; it does not change the host driver. |
| One-device/one-process error | Expose exactly one GPU or MIG slice before launch and use `WORLD_SIZE=1`; do not launch with torchrun/accelerate distributed mode. |
| CUDA out of memory | Use an available slice and the small preset; check current memory use. Reduce budgets deliberately if needed and restart the process after failure. Full policies/results are intentionally retained. |
| Wrong torch/transformers/TRL/PEFT version | Restore the version contract with setup or the recorded local freeze. A local CUDA suffix is permitted by the training version check. |
| AgentTune Git-origin mismatch | Install the exact pinned Git requirement. A matching PyPI version or local source folder does not satisfy the installed commit-origin check. |
| Missing `environment_factory` | Installed TRL is incompatible with the integration; restore the pinned stack. |
| Prompt exceeds context | Keep policy/ticket intact. Increase context only within actual model support, or reduce completion reserve with an explicit config change. |
| Configured context exceeds model support | Lower `context_limit` or select a compatible model; arbitrary larger context is not enabled by the config. |
| No fitting failures | The selected baseline sample produced no repair tasks. Increase `fit_scan_tasks`; no synthetic failure dataset is substituted. |
| All observed baseline groups tie | Sparse terminal rewards may give no useful GRPO reward advantage. Inspect traces and consider a stronger model or larger rollout budget. This warning is not a successful learning result. |
| Gate rejects unchanged weights | Inspect training logs and group reward variance; an unchanged adapter is not promoted. |
| Gate rejects no improvement | Strict aggregate validation improvement is required even if the inherited tolerance would permit a tie. Rejection can be a valid completed run. |
| Missing evaluated simulation | Gym returned no terminal evaluation; inspect the exception/log and full run. Infrastructure errors are not silently scored as failures. |
| Reward routing/task alignment error | Completion, environment and task IDs do not match the required contract. Restore pinned interfaces or fix the harness before interpreting scores. |
| Unknown/held-out trajectory in training batch | Failure provenance or partition checks rejected the batch. Investigate the data flow; do not weaken leakage checks to continue. |
| Four-hour wait timeout | Training/gate exceeded the source-defined wait budget. Ensure the process/worker has stopped before retrying. No automatic optimizer resume exists. |
| Retraining succeeds but gate callback missing | Inspect AgentTune logs; the run treats this as an incomplete cycle and raises. |
| Saved weights differ / reload tensor differs | Artifact integrity verification failed. Do not claim the adapter was verified or selected successfully. Preserve the files for diagnosis. |
| `final_report.json` exists but process failed | Reload is after report creation. Check exit status and reload logs to determine the last successful stage. |
| Final artifacts absent after smoke | Expected: smoke exits after one evaluated Gym episode and never trains. |

## Useful diagnostics

```bash
.venv/bin/tau2-grpo doctor
uv pip check --python .venv/bin/python
nvidia-smi
.venv/bin/python -c 'import tau2; print(tau2.__file__)'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

When CUDA is available, this tiny check exercises an allocation and reduction
without loading a model:

```bash
.venv/bin/python - <<'PYCODE'
import torch
print(torch.ones(16, device="cuda").sum().item())
print("free/total GiB:", [round(n / 2**30, 2) for n in torch.cuda.mem_get_info()])
PYCODE
```

`doctor` is a lightweight metadata/CUDA check, not a guarantee that all model
imports, native kernels or memory-intensive rollouts will work. The standalone
project does not ship an inference server, automatic dependency upgrades,
cooperative worker cancellation or multi-round recovery automation.
