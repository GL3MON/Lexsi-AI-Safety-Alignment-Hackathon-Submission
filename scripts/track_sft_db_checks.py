"""Track failed database checks in SFT source episodes without filtering the data."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "outputs/retail-data/retail-sft-final/sft.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output_dir or dataset.parent
    output.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    rows_by_source = defaultdict(list)
    for line_number, line in enumerate(dataset.read_text().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            rows_by_source[row["source_episode"]].append((line_number, row))

    tasks = defaultdict(lambda: {"source_trajectories": 0, "db_failed": 0, "db_passed": 0, "db_unknown": 0})
    failures, unknown = [], []
    for source, rows in sorted(rows_by_source.items()):
        episode = json.loads(Path(source).read_text())
        record = episode["record"]
        task_id = str(record["task_id"])
        if any(str(row["task_id"]) != task_id for _, row in rows):
            raise ValueError("Dataset/source task mismatch: " + source)
        info = record.get("reward_info") or (episode.get("simulation") or {}).get("reward_info") or {}
        db = info.get("db_check") or {}
        value = db.get("db_reward")
        status = "db_failed" if value == 0 else "db_passed" if value == 1 else "db_unknown"
        tasks[task_id]["source_trajectories"] += 1
        tasks[task_id][status] += 1
        entry = {
            "task_id": task_id,
            "source_episode": source,
            "source_reward": record.get("reward"),
            "db_check": db or None,
            "reward_breakdown": info.get("reward_breakdown"),
            "sft_line_numbers": [number for number, _ in rows],
            "sft_example_ids": [row.get("id") for _, row in rows],
            "failed_expected_write_actions": [
                check["action"] for check in info.get("action_checks") or []
                if check.get("tool_type") == "write" and check.get("action_match") is False
            ],
        }
        if status == "db_failed":
            failures.append(entry)
        elif status == "db_unknown":
            unknown.append(entry)

    sort_key = lambda value: (0, int(value)) if value.isdigit() else (1, value)
    ids = sorted({entry["task_id"] for entry in failures}, key=sort_key)
    report = {
        "dataset": str(dataset), "dataset_sha256": original_hash,
        "scope": "training SFT source episodes only; validation is not included",
        "tracking_only": True, "dataset_modified": False,
        "note": "Flags individual generations. A flagged task may also have passing generations. Missing DB checks are tracked separately, never treated as failures.",
        "source_trajectories": len(rows_by_source),
        "db_failed_source_trajectories": len(failures),
        "db_unknown_source_trajectories": len(unknown),
        "db_failed_task_ids": ids,
        "per_task": {key: tasks[key] for key in sorted(tasks, key=sort_key)},
        "failed_trajectories": failures,
        "unknown_db_check_trajectories": unknown,
    }
    (output / "db-check-tracker.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "db-check-failed-task-ids.txt").write_text("\n".join(ids) + "\n")
    with (output / "db-check-tracker.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "source_reward", "db_reward", "sft_line_numbers", "source_episode"])
        writer.writeheader()
        for entry in failures:
            writer.writerow({
                "task_id": entry["task_id"], "source_reward": entry["source_reward"],
                "db_reward": entry["db_check"]["db_reward"],
                "sft_line_numbers": ",".join(map(str, entry["sft_line_numbers"])),
                "source_episode": entry["source_episode"],
            })
    text = [
        "# Retail SFT database-check tracker", "",
        f"{len(failures)} of {len(rows_by_source)} training source trajectories failed their DB check, across {len(ids)} task IDs.",
        "Tracking only: all SFT examples remain unchanged. Flags apply to individual generations, not every generation of that task.",
        f"{len(unknown)} source trajectories have no DB check and are listed separately in the JSON tracker.", "",
        "| Task ID | Failed DB | Passed DB | No DB check | Total source trajectories |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task_id in ids:
        t = tasks[task_id]
        text.append(f"| {task_id} | {t['db_failed']} | {t['db_passed']} | {t['db_unknown']} | {t['source_trajectories']} |")
    text += ["", "## Affected source episodes", ""]
    for entry in failures:
        source = entry["source_episode"]
        text.append(f"- Task {entry['task_id']}, SFT lines {entry['sft_line_numbers']}: [source episode]({source})")
    text += ["", f"Dataset SHA-256: `{original_hash}`", ""]
    (output / "db-check-tracker.md").write_text("\n".join(text))
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == original_hash
    print(json.dumps({"failed_source_trajectories": len(failures), "affected_tasks": len(ids), "task_ids": ids, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
