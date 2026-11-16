"""Check native calls, a retail choice and long-context lookup without executing tools."""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def benchmark(base_url):
    cases = [
        (
            "order",
            "Look up the exact order ID '#W12345', including its leading #.",
            "get_order_details",
            {"order_id": "#W12345"},
            {
                "order_id": {
                    "type": "string",
                    "description": "Exact order ID, including its leading # symbol. Do not remove or change characters.",
                }
            },
        ),
        (
            "retail_choice",
            "Choose the cheapest AVAILABLE i7-or-better laptop. Choices: A is i7, $100, unavailable; B is i7, $130, available; C is i9, $160, available; D is i5, $90, available. Call select_item with the chosen item_id.",
            "select_item",
            {"item_id": "B"},
            {"item_id": {"type": "string"}},
        ),
        (
            "long_context",
            "The verified customer order ID is #W5838674.\n"
            + "Historical store note: records in this paragraph are irrelevant to the current order.\n"
            * 450
            + "\nLook up the verified customer order ID given at the beginning. Preserve its leading # symbol.",
            "get_order_details",
            {"order_id": "#W5838674"},
            {
                "order_id": {
                    "type": "string",
                    "description": "Exact order ID, including its leading # symbol. Do not remove or change characters.",
                }
            },
        ),
    ]
    real_episode = (
        Path(__file__).resolve().parents[1]
        / "outputs/retail-data/qwen-sft-fixed/runs/run-a42d37dd2a/episode-0001.json"
    )
    replay = None
    if real_episode.exists():
        replay = json.loads(real_episode.read_text())
        target_index = max(
            i
            for i, message in enumerate(replay["messages"])
            if any(
                call["function"]["name"] == "exchange_delivered_order_items"
                for call in message.get("tool_calls", [])
            )
        )
        target = replay["messages"][target_index]["tool_calls"][0]["function"]
        cases.append(
            (
                "retail_exchange_replay",
                None,
                target["name"],
                json.loads(target["arguments"]),
                None,
            )
        )
    results = []
    for index, (name, prompt, function, expected, properties) in enumerate(
        [cases[0]] + cases
    ):
        body = {
            "model": "qwen3.5-4b",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": "This is an isolated tool-format test. Make exactly the requested tool call. Do not add explanations or prose. Use only supplied facts. Preserve identifiers exactly, including # symbols.",
                },
                {"role": "user", "content": prompt},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": function,
                        "description": "Execute the requested lookup or item selection.",
                        "parameters": {
                            "type": "object",
                            "properties": properties or {},
                            "required": list(properties or {}),
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if prompt is None:
            body["messages"] = [
                {
                    key: value
                    for key, value in message.items()
                    if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
                }
                for message in replay["messages"][:target_index]
            ]
            body["messages"][0]["content"] += (
                "\nFor this isolated replay, emit the next tool call without additional prose. Preserve exact identifiers."
            )
            body["tools"] = replay["record"]["tools"]
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        calls = result["choices"][0]["message"].get("tool_calls") or []
        passed = (
            len(calls) == 1
            and calls[0]["function"]["name"] == function
            and json.loads(calls[0]["function"]["arguments"]) == expected
        )
        if index:
            results.append(
                {
                    "case": name,
                    "passed": passed,
                    "seconds": elapsed,
                    "usage": result.get("usage"),
                    "response": result,
                }
            )
    return {
        "passed": all(row["passed"] for row in results),
        "cases": results,
        "median_seconds": statistics.median(row["seconds"] for row in results),
        "total_seconds": sum(row["seconds"] for row in results),
        "output_tokens_per_second": sum(
            row["usage"]["completion_tokens"] for row in results
        )
        / sum(row["seconds"] for row in results),
        "note": "Small formatting/selection/long-context smoke, not a full retail quality evaluation. No tools executed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "cases"}, indent=2
        )
    )
