# What went wrong with the prompts?

The main bug was in my code for creating extra Glaive RL examples. Some examples
showed the model **two different lists of available tools** when there should
have been one. The dataset's expected call was still present, but the prompt
used to teach it was inconsistent.

This does not mean every failed call was caused by that bug. It does mean the
old results were a poor basis for deciding that the model cannot follow hints.

## 1. A simple example

Imagine we want to teach the model this interaction:

```text
CUSTOMER: Please help reset my password. Use the email returned by the lookup.

ASSISTANT: resolve_reference_email()

TOOL RESPONSE: The email is mira@example.org.

EXPECTED NEXT CALL:
generate_password_reset_token(email="mira@example.org")
```

This is a synthetic fixture: the lookup result is supplied by our test data,
not obtained from a real password-reset service.

The prompt needs to list the original password-reset tool and the additional
lookup tool. This simplified version shows the intended structure; the actual
catalog contains full JSON schemas.

```text
Available tools:
<tools>
[
  generate_password_reset_token(email),
  resolve_reference_email()
]
</tools>
```

## 2. What the buggy generator actually did

The original Glaive system text mentioned the delimiters in a documentation
sentence, then included the real tool catalog later:

```text
Function signatures are provided within <tools></tools> XML tags.

Here are the available tools:
<tools>
[
  generate_password_reset_token(email)
]
</tools>
```

My code searched for the **first** opening and closing tools tags and replaced
what was between them. It accidentally replaced the empty documentation example
instead of the real catalog.

After adding the synthetic lookup tool, the prompt therefore looked like this:

```text
Function signatures are provided within
<tools>
[
  generate_password_reset_token(email),
  resolve_reference_email()
]
</tools>
XML tags.

Here are the available tools:
<tools>
[
  generate_password_reset_token(email)
]
</tools>
```

Now the model sees an expanded list first, followed by a stale list. The supplied
history uses `resolve_reference_email`, but the later list does not include it.
That ambiguity could confuse which tools and observations the model should use.
It is a plausible source of trouble, not proof that it caused every failure.

**405 synthetic tool-result-copy examples had this duplicate-catalog bug.**

## 3. A second, smaller issue in the tokenizer template

The chat template also included this explanatory sentence:

```text
Use the functions listed in <tools> when they are needed ...
```

That is an unmatched opening tag in prose. It is **not a second JSON catalog**.
However, it makes the final rendered prompt's markup ambiguous and causes strict
checks to find two opening tags but only one closing tag.

The clean template now says:

```text
Use the functions in the available tool catalog when they are needed ...
```

The actual catalog still uses one opening and one closing tag.

## 4. What is fixed now?

The generator builds the system prompt from the actual tool schemas instead of
editing the first matching tags in old prose. The clean template removes the
stray documentation tag. Checks verify:

- Exactly one tool catalog in the system prompt and the final rendered prompt.
- The catalog matches the tools stored with that example.
- The expected calls satisfy the verifier and schema checks.
- The chat template can render the conversation roles correctly.

All **2,107 corrected data rows** passed these rendering and label checks.
Clarification rows and customer follow-ups are connected by the runner, leaving
**1,702 episode starts**.

Corrected data: `outputs/glaive-rl-signals-v2/train.jsonl`.

The original data and historical checkpoints are retained for comparison. Fixing
these files does not change the weights of an already-trained checkpoint.

## 5. Does this mean the SFT was broken?

**This particular duplicate-catalog bug was introduced when generating the extra
RL examples.** The audit found no duplicate JSON catalogs in the 2,369 Glaive SFT
training examples or 220 SFT validation examples.

There is no reason to repeat SFT solely because of this bug. We are repeating RL
from the pre-RL SFT checkpoint-475 using the corrected prompts. This conclusion
does not certify that every other aspect of the SFT is perfect.

## 6. What about the hints?

That was a separate problem. Some earlier hints were long descriptions of what
went wrong. Some proposed re-asking for information already supplied. The teacher
prompt also asked the model to evaluate an earlier action, which could lead it
to discuss the mistake instead of performing the next action.

For a clean capability check, we used direct manual instructions instead:

```text
Ask the customer for the missing email before calling a lookup tool.
```

```text
Copy the email from the latest customer reply exactly.
```

We tested 20 fixed cases, each with two seeds, on the frozen RL checkpoint-16.
The exact rendered prompts were saved. This test used simple ASCII values and
no translation tasks.

| Check | No manual hint | With manual hint |
|---|---:|---:|
| Ask for missing information | 3/16 | 16/16 |
| Copy supplied values | 10/12 | 11/12 |
| Use customer corrections | 12/12 | 12/12 |

We inspected all 16 hinted missing-information responses: each actually asked
for the needed field without making a tool call.

**The model can follow clear hints in this direct-instruction setup.** This test
changed several prompt factors together, so it does not isolate the contribution
of each fix. It also does not prove that Qwen can generate good hints autonomously,
or that the paper's distillation method will improve hint-free behavior.

## 7. What is running now?

The fresh job uses the reserved **Glaive RL split**, not Tau2 tasks:

- Start from Glaive SFT checkpoint-475, not the previous RL checkpoint.
- Use the corrected RL data and cleaned template.
- Run 100 groups with 8 rollouts each on MIG0.
- Train with the existing verifier rewards and GRPO, without teacher hints.
- Log samples, rewards, and per-rollout advantages to W&B.

The old 487 reserved Glaive RL prompts supplied the source for the expanded
training data. Their groups remain separate from SFT train, SFT validation, and
test. The fresh run has already completed its first optimizer update.

## Files to inspect

- [Audit results](../outputs/diagnostics/clean-manual-hint-baseline/audit.json)
- [Manual-hint conversations](../outputs/diagnostics/clean-manual-hint-baseline/conversations.txt)
- [Exact rendered prompts](../outputs/diagnostics/clean-manual-hint-baseline/rendered/)
- [Corrected RL configuration](../configs/rl-glaive.json)
- [Fresh training log](../outputs/logs/glaive-rl-clean-v2-training.log)
