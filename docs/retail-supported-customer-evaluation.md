# Supported-request customer variant

The customer-only overlay in `templates/retail_customer_supported_requests.txt` limits development conversations to supported retail actions. It does not change tool schemas, tool execution, the agent prompt, or the training data.

For task 63, the original request to remove an expensive speaker is replaced by a request to find the cheapest available Bluetooth speaker variant below $300 and replace the existing speaker one for one. The rest of the order must be retained. Requests to guess a poem are omitted. The existing controlled email-available overlay can be combined with this overlay.

This changes the original task and must be labeled as a controlled diagnostic, not an unchanged Tau2 benchmark result. The customer receives no reference action sequence, hidden expected prices, or correct replacement identifier. It must learn those details from the agent. The agent still needs to retrieve inventory, select the correct product variant, obtain confirmation, execute the modification, and report the resulting total.

The overlay is opt-in through the evaluator's `--user-prompt-file`; original Tau2 task files and default customer behavior remain intact.
