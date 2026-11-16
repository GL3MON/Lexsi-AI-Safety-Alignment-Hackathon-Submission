import copy
import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "memory_experiment",
    Path(__file__).resolve().parents[1] / "scripts/test_retail_summarizer.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class MemoryValidationTests(unittest.TestCase):
    def fixture(self):
        prefix = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "return my item"},
            {"role": "assistant", "content": "lookup"},
            {
                "role": "tool",
                "content": json.dumps(
                    {"name": "find_user_id_by_email", "content": "user_1"}
                ),
            },
            {"role": "assistant", "content": "inspect order"},
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "name": "get_order_details",
                        "content": {
                            "order_id": "#W1",
                            "user_id": "user_1",
                            "status": "delivered",
                            "items": [
                                {
                                    "item_id": "item_1",
                                    "product_id": "product_1",
                                    "price": 2,
                                    "name": "Item",
                                    "options": {},
                                }
                            ],
                            "payment_history": [{"payment_method_id": "card_1"}],
                        },
                    }
                ),
            },
            {
                "role": "assistant",
                "content": "Return item_1 from #W1 to card_1. Confirm?",
            },
            {"role": "user", "content": "Yes"},
        ]
        memory = {
            "customer": {"user_id": "user_1", "authenticated": True, "source_ids": [3]},
            "active_order": {
                "order_id": "#W1",
                "status": "delivered",
                "source_ids": [5],
            },
            "items": [
                {
                    "item_id": "item_1",
                    "product_id": "product_1",
                    "price": 2,
                    "name": "Item",
                    "options": {},
                    "source_ids": [5],
                }
            ],
            "replacement_items": [],
            "payment": {
                "payment_method_id": "card_1",
                "balance": None,
                "source_ids": [5],
            },
            "confirmation": {
                "status": "confirmed",
                "order_id": "#W1",
                "item_ids": ["item_1"],
                "source_ids": [6, 7],
            },
            "pending_action": {"executed": False, "source_ids": [7]},
            "preferences": [],
            "resolved_requests": [],
            "actions_taken": [],
        }
        expected = {
            "name": "return_delivered_order_items",
            "arguments": {
                "order_id": "#W1",
                "item_ids": ["item_1"],
                "payment_method_id": "card_1",
            },
        }
        return memory, prefix, expected

    def test_verified_sources_pass(self):
        memory, prefix, expected = self.fixture()
        self.assertEqual(experiment.validate_memory(memory, prefix, expected, 6), [])

    def test_wrong_values_confirmation_future_and_unexecuted_actions_are_detected(self):
        original, prefix, expected = self.fixture()
        mutations = [
            lambda m: m["items"][0].update(price=999),
            lambda m: m["items"][0].update(name="Invented name"),
            lambda m: m["confirmation"].update(source_ids=[7]),
            lambda m: m["pending_action"].update(executed=True),
            lambda m: m["pending_action"].update(source_ids=[8]),
            lambda m: m["payment"].update(balance=100),
            lambda m: m.update(
                actions_taken=[
                    {
                        "name": "return_delivered_order_items",
                        "status": "succeeded",
                        "source_ids": [6],
                    }
                ]
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                memory = copy.deepcopy(original)
                mutate(memory)
                self.assertTrue(experiment.validate_memory(memory, prefix, expected, 6))
