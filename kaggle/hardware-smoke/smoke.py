"""Private Kaggle hardware/FP16 smoke; no model training or credential output."""
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path('/kaggle/working')
ROOT.mkdir(parents=True, exist_ok=True)
report = {'stage': 'starting', 'python': platform.python_version(),
          'purpose': 'Verify two-GPU allocation, FP16 kernels, internet, and installed packages.'}


def save():
    (ROOT / 'hardware-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


try:
    report['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    report['packages'] = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'wandb', 'vllm', 'triton', 'bitsandbytes']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    import torch
    report['cuda_version'] = torch.version.cuda
    report['gpu_count'] = torch.cuda.device_count()
    if report['gpu_count'] < 2:
        raise RuntimeError('The requested two-GPU layout needs two allocated GPUs')
    report['devices'] = []
    for index in range(report['gpu_count']):
        with torch.cuda.device(index):
            prop = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            item = {'index': index, 'name': prop.name, 'total_gib': total / 2**30,
                    'free_gib': free / 2**30, 'compute_capability': [prop.major, prop.minor],
                    'native_bf16': torch.cuda.is_bf16_supported(including_emulation=False)}
            device = torch.device('cuda', index)
            torch.manual_seed(42)
            start = time.monotonic()
            if index == 0:
                with torch.inference_mode():
                    x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                    y = x @ x.T
                    assert torch.isfinite(y).all().item()
                item['fp16_inference'] = 'passed'
                del x, y
            else:
                # Exercise mixed-precision forward/backward and an actual optimizer update.
                model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(),
                                            torch.nn.Linear(4096, 1024)).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler('cuda')
                before = model[0].weight.detach().clone()
                x = torch.randn(8, 1024, device=device)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(x).float().square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                assert torch.isfinite(loss).item()
                assert not torch.equal(before, model[0].weight)
                item.update(fp16_optimizer_update='passed', loss=float(loss.detach()))
                del x, before, optimizer, scaler, model, loss
            torch.cuda.synchronize(index)
            item['smoke_seconds'] = time.monotonic() - start
            item['peak_allocated_mib'] = torch.cuda.max_memory_allocated(index) / 2**20
            torch.cuda.empty_cache()
            report['devices'].append(item)
    report['model_metadata_access'] = {}
    for model in ['CohereLabs/tiny-aya-global', 'Qwen/Qwen3.5-4B']:
        try:
            with urllib.request.urlopen('https://huggingface.co/' + model + '/resolve/main/config.json', timeout=30) as response:
                config = json.load(response)
            report['model_metadata_access'][model] = {'accessible': True, 'model_type': config.get('model_type')}
        except Exception as error:
            report['model_metadata_access'][model] = {'accessible': False, 'error_type': type(error).__name__}
    report['stage'] = 'passed'
except Exception as error:
    report.update(stage='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    save()
    raise
save()
