"""Shared causal retail memory provenance checks used by SFT and RL."""

import json
import re

def contains(value, text):
    return bool(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
    )

def summary_errors(memory, prefix):
    """Reject future citations and novel identifier/numeric database records."""
    errors = []
    source = {i: m for i, m in enumerate(prefix)}

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if "source_ids" in value:
                ids = value["source_ids"]
                no_claim = all(
                    field is None or field is False or field == [] or field == {}
                    for key, field in value.items() if key != "source_ids"
                )
                if ids == [] and no_claim:
                    pass  # Null/unknown records make no factual assertion to cite.
                elif (
                    not isinstance(ids, list)
                    or not ids
                    or any(type(i) is not int or i < 1 or i >= len(prefix) for i in ids)
                ):
                    errors.append("invalid_source_ids")
                else:
                    text = "\n".join(source[i].get("content") or "" for i in ids)
                    tool_bodies = []
                    for index in ids:
                        if source[index]["role"] == "tool":
                            tool_bodies.append(json.loads(source[index]["content"]))
                            # A response such as "Error: Order not found" refers
                            # to the ID in its paired call, not an echoed result.
                            if index and prefix[index - 1].get("tool_calls"):
                                text += "\n" + json.dumps(
                                    prefix[index - 1]["tool_calls"]
                                )
                    if (
                        value.get("order_id") is not None
                        and value.get("status") is not None
                    ):
                        statuses = []
                        for index in ids:
                            if source[index]["role"] != "tool":
                                continue
                            body = json.loads(source[index]["content"]).get("content")
                            if (
                                isinstance(body, dict)
                                and body.get("order_id") == value["order_id"]
                            ):
                                statuses.append(body.get("status"))
                            if (
                                isinstance(body, str)
                                and body.lower().startswith("error")
                                and index
                            ):
                                for call in prefix[index - 1].get("tool_calls", []):
                                    arguments = call["function"]["arguments"]
                                    arguments = (
                                        json.loads(arguments)
                                        if isinstance(arguments, str)
                                        else arguments
                                    )
                                    if arguments.get("order_id") == value["order_id"]:
                                        statuses += [
                                            "not_found",
                                            "not found",
                                            "error",
                                            "invalid",
                                            "unknown",
                                        ]
                        if value["status"] not in statuses:
                            errors.append("order_status_does_not_match_cited_result")
                    if value.get("authenticated") is True and not any(
                        body.get("name")
                        in {"find_user_id_by_email", "find_user_id_by_name_zip"}
                        and body.get("content") == value.get("user_id")
                        for body in tool_bodies
                    ):
                        errors.append("authentication_without_lookup_result")
                    if value.get("item_id") is not None:
                        candidates = []
                        for wrapper in tool_bodies:
                            body = wrapper.get("content")
                            if not isinstance(body, dict):
                                continue
                            candidates += [
                                item
                                for item in body.get("items", [])
                                if item.get("item_id") == value["item_id"]
                            ]
                            candidates += [
                                dict(
                                    item,
                                    product_id=body.get("product_id"),
                                    name=body.get("name"),
                                )
                                for item in body.get("variants", {}).values()
                                if item.get("item_id") == value["item_id"]
                            ]
                        if not any(
                            all(
                                item.get(k) == value[k]
                                for k in [
                                    "item_id",
                                    "product_id",
                                    "price",
                                    "name",
                                    "available",
                                ]
                                if k in value and value[k] is not None
                            )
                            and all(
                                {
                                    key.replace("_", " ").casefold(): val
                                    for key, val in item.get("options", {}).items()
                                }.get(k.replace("_", " ").casefold())
                                == v
                                for k, v in value.get("options", {}).items()
                            )
                            for item in candidates
                        ):
                            errors.append(
                                "item_fields_do_not_match_cited_item:"
                                + str(value["item_id"])
                            )
                    for key, field in value.items():
                        if field is None or key == "source_ids":
                            continue
                        if key.endswith("_id") and not contains(str(field), text):
                            errors.append("unsupported_field:" + key + "=" + str(field))
                        if key in {"price", "balance"} and isinstance(
                            field, (float, int)
                        ):
                            candidates = [
                                float(v)
                                for v in re.findall(
                                    r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text
                                )
                            ]
                            if not any(abs(field - v) < 0.00001 for v in candidates):
                                errors.append("unsupported_numeric:" + key)
            for child in value.values():
                visit(child)

    visit(memory)
    return sorted(set(errors))

def controller_history(prefix):
    """Retain mentioned order IDs and actual failed calls without claiming validity."""
    orders = {}
    errors = []
    for index, message in enumerate(prefix):
        if message["role"] in {"user", "tool"}:
            for order_id in re.findall(r"#W\d+", message.get("content") or ""):
                orders.setdefault(order_id, []).append(index)
        if message["role"] == "tool":
            payload = json.loads(message["content"])
            body = payload.get("content")
            failed = (
                isinstance(body, str)
                and body.lstrip().lower().startswith("error")
                or isinstance(body, dict)
                and bool(body.get("error"))
            )
            if failed and index and prefix[index - 1].get("tool_calls"):
                errors.append(
                    {
                        "source_ids": [index - 1, index],
                        "status": "failed",
                        "calls": prefix[index - 1]["tool_calls"],
                        "observation": body,
                    }
                )
    return {
        "mentioned_orders": [
            {"order_id": key, "source_ids": ids} for key, ids in orders.items()
        ],
        "order_ids_are_not_proof_of_validity": True,
        "failed_calls": errors,
        "owner": "controller",
    }
