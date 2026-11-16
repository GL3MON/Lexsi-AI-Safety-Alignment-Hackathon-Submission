import unittest

from tau2_grpo.retail_actor_prompts import customer_prompt, lookup_recovery_instruction


class ActorPromptTests(unittest.TestCase):
    def test_recovery_uses_known_identity_zip_not_destination_zip(self):
        scenario = {"instructions": {
            "known_info": "Yusuf Taylor lives in ZIP 95154.",
            "reason_for_call": "Change shipping ZIP to 62701.",
            "unknown_info": "Does not remember email.",
        }}
        text = lookup_recovery_instruction(scenario)
        self.assertIn("95155", text)
        self.assertIn("95154", text)
        self.assertNotIn("62701", text)

    def test_unknown_email_is_not_exposed_as_known_contact(self):
        text = lookup_recovery_instruction({"instructions": {
            "known_info": "Email possible@example.com, ZIP 32255.",
            "unknown_info": "You do not remember your email.",
        }})
        self.assertNotIn("possible@example.com", text)
        self.assertIn("32256", text)
        self.assertIn("32255", text)

    def test_missing_known_identity_does_not_generate_a_fake_value(self):
        text = lookup_recovery_instruction({"instructions": {"known_info": "Name only"}})
        self.assertNotIn("CONCRETE ONE-TYPO", text)
        self.assertIn("no fictional", text)

    def test_default_prompt_preserves_task_and_adds_identity_rules(self):
        self.assertTrue(customer_prompt("Original task").startswith("Original task\n\n"))
        self.assertIn("Existing task identities must remain unchanged",customer_prompt("Original task"))
        with self.assertRaises(ValueError):
            customer_prompt("Original task", profile="invalid")


if __name__ == "__main__":
    unittest.main()
