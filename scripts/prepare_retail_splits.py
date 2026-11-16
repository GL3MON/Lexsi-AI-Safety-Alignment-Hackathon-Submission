"""Freeze disjoint retail SFT/RL/validation tasks while preserving the official test."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    from loguru import logger

    logger.remove()
    from tau2.domains.retail.environment import get_tasks, get_tasks_split
    from tau2.domains.retail.utils import RETAIL_TASK_SET_PATH

    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-tasks", type=int, default=40)
    p.add_argument("--rl-tasks", type=int, default=22)
    p.add_argument("--validation-tasks", type=int, default=10)
    p.add_argument(
        "--output", type=Path, default=root / "outputs/retail-data/split_manifest.json"
    )
    args = p.parse_args()
    if min(args.sft_tasks, args.rl_tasks, args.validation_tasks) < 1:
        p.error("Partition sizes must be positive")
    if args.output.exists():
        p.error(f"Refusing to replace an existing manifest: {args.output}")
    official = get_tasks_split()
    tasks = {t.id: t for t in get_tasks(None)}
    train, test = set(official["train"]), set(official["test"])
    assert train.isdisjoint(test)
    issues = {
        i: [
            x.model_dump(mode="json")
            for x in t.issues or []
            if x.status.value == "open"
        ]
        for i, t in tasks.items()
    }
    quarantined = sorted(i for i in train if issues[i])
    eligible = sorted(train - set(quarantined))
    if len(eligible) != args.sft_tasks + args.rl_tasks + args.validation_tasks:
        p.error(f"Sizes must sum to {len(eligible)} eligible training tasks")
    # Exploratory retail previews stay out of the new validation set.
    observed = set()
    for pattern in (
        "outputs/tau2-reward-analysis/run-*/settings.json",
        "outputs/run-*/runtime.json",
    ):
        for path in root.glob(pattern):
            data = json.loads(path.read_text())
            if data.get("domain") == "retail":
                observed.update(str(i) for i in data.get("task_ids", []))
                if data.get("task_id") is not None:
                    observed.add(str(data["task_id"]))
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    validation = [i for i in eligible if i not in observed][: args.validation_tasks]
    if len(validation) != args.validation_tasks:
        p.error("Not enough unpreviewed validation tasks")
    fit = [i for i in eligible if i not in set(validation)]
    sft, rl = fit[: args.sft_tasks], fit[args.sft_tasks :]
    partitions = {
        "sft": sft,
        "rl": rl,
        "validation": validation,
        "test": sorted(test),
        "quarantine": quarantined,
    }
    values = list(partitions.values())
    for index, ids in enumerate(values):
        assert all(set(ids).isdisjoint(other) for other in values[index + 1 :])
    assert set(sft + rl + validation + quarantined) == train
    manifest = {
        "domain": "retail",
        "seed": args.seed,
        "split_unit": "task_id",
        "task_file": str(RETAIL_TASK_SET_PATH),
        "task_file_sha256": hashlib.sha256(
            Path(RETAIL_TASK_SET_PATH).read_bytes()
        ).hexdigest(),
        "official_counts": {
            "train": len(train),
            "test": len(test),
            "total": len(tasks),
        },
        "counts": {k: len(v) for k, v in partitions.items()},
        "partitions": partitions,
        "previously_previewed_task_ids": sorted(observed),
        "open_issues": {i: v for i, v in issues.items() if v},
        "trajectory_plan": {
            "accepted_sft_trajectories_per_task": [3, 5],
            "accepted_sft_total": [3 * len(sft), 5 * len(sft)],
            "rl_rollouts_per_task_per_sweep": [4, 8],
            "rl_rollouts_per_sweep": [4 * len(rl), 8 * len(rl)],
        },
        "notes": [
            "Partitions are disjoint by task ID; this does not assert disjoint customers or orders.",
            "Quarantined training tasks require review before use. Official test tasks, including issue flags, are preserved.",
            "SFT requires verified successful policy-correct multi-turn trajectories; task reference actions alone are not conversations.",
            "RL reuses tasks with fresh rollouts; rollout counts are not independent benchmark tasks.",
            "Do not use validation or test tasks to generate training data.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(manifest, file, indent=2)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "counts": manifest["counts"],
                "trajectory_plan": manifest["trajectory_plan"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
