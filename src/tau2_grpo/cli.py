"""CLI entry point; heavy dependencies are imported only when needed."""

import argparse
import json
import os
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

from tau2_grpo.config import load_config

PROJECT = Path(__file__).resolve().parents[2]


def main():
    """Validate settings, check the machine, smoke-test Gym, or train locally."""
    if len(sys.argv) > 1 and sys.argv[1] == "retail-rl":
        from tau2_grpo.rl import main as retail_rl_main
        return retail_rl_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["config", "doctor", "smoke", "preview", "train"])
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/local.json")
    parser.add_argument("--tau2", type=Path, default=PROJECT.parent / "tau2-bench")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "config":
        print(json.dumps(asdict(cfg), indent=2))
        return
    if cfg.domain == "retail" and args.command not in {"preview", "doctor"}:
        parser.error("Retail currently supports preview, not smoke or training")
    if not (args.tau2 / "src/tau2").is_dir():
        parser.error(f"Missing tau2 checkout: {args.tau2}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.command == "doctor":
        from importlib.metadata import PackageNotFoundError, version

        ok = True
        for name in [
            "tau2",
            "aligntune",
            "torch",
            "transformers",
            "trl",
            "peft",
            "gymnasium",
        ]:
            try:
                print(f"{name}: {version(name)}")
            except PackageNotFoundError:
                print(f"{name}: MISSING")
                ok = False
        try:
            import torch

            print("CUDA available:", torch.cuda.is_available())
            ok = ok and torch.cuda.is_available()
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
        except ImportError:
            ok = False
        raise SystemExit(0 if ok else 1)
    os.environ.update(
        TAU2_CHECKOUT=str(args.tau2.resolve()),
        TAU2_GRPO_CONFIG=str(args.config.resolve()),
        TAU2_GRPO_OUTPUT=str(args.output.resolve()),
        TAU2_GRPO_SMOKE="1" if args.command == "smoke" else "0",
        TAU2_GRPO_PREVIEW="1" if args.command == "preview" else "0",
    )
    if cfg.domain == "retail":
        from tau2_grpo.retail_preview import run_retail_preview

        run_retail_preview(cfg, args.output)
        return
    # tau2 locates its data relative to its installed editable checkout.
    runpy.run_module("tau2_grpo.experiment", run_name="__main__")


if __name__ == "__main__":
    main()
