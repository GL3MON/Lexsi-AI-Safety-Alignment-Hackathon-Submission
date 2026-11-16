"""Versioned customer scenarios for corrective retail demonstrations."""

import re

IDENTITY_STYLE_INSTRUCTION = (
    "IDENTITY VALUES: Use the exact identity supplied by the task or synthetic fixture. "
    "Never substitute generic identities such as John Doe, Jane Doe, Sara Doe, "
    "john.doe@example.com, sara@example.com, or user@example.com. "
    "For new synthetic identities only, use varied plausible names and distinctive "
    "email local parts, preserving dots, plus tags, case and supplied digits exactly; "
    "avoid one repeated naming pattern. If identity is unknown, say so rather than "
    "inventing it. Existing task identities must remain unchanged, even if generic."
)

CUSTOMER_PROFILES = {
    "standard": "",
    "identity-clarification": (
        "SCENARIO: MISSING IDENTITY AND ORDER DISCOVERY. Start with only your request. "
        "Wait for the agent to ask for identity. Provide only known task facts when asked. "
        "If email is unknown, say so and provide your actual known name/ZIP. "
        "Let the agent retrieve account orders instead of inventing identifiers."
    ),
    "lookup-recovery": (
        "SCENARIO: ONE CORRECTABLE IDENTITY LOOKUP FAILURE. Start with only your request. "
        "On the FIRST occasion you provide a known identity field, make exactly ONE controlled typo: "
        "if you know an email, insert one extra x before its @; otherwise, if your original task gives "
        "a ZIP, change only its last digit to the next digit (9 becomes 0). Keep the original known "
        "name unchanged. Never invent an email if the task says you do not remember it. "
        "After the agent reports the lookup failed or asks you to verify it, explicitly apologize "
        "for that single typo and give the exact ORIGINAL task value. Never make another typo or "
        "invent any order IDs. If neither email nor ZIP is known, do not fabricate a value; "
        "ask for another supported identification method. Original task-specific constraints still apply."
    ),
    "confirmation-execution": (
        "SCENARIO: CLEAR CONSENT AND VERIFIED COMPLETION REPORTING. Start with only your request "
        "and answer identity questions truthfully. Follow any original task instruction to change "
        "the request when first asked for confirmation. Once the agent explains the correct updated "
        "plan, give a direct yes for the precise scope. Do not add a new objection or repeat the "
        "same confirmation indefinitely. Wait for execution and a specific result before ending. "
        "If the agent asks the same unchanged confirmation again, say you already approved it. "
        "Do not assert imaginary refunds, order updates, or other successful outcomes."
    ),
}


def customer_prompt(base_prompt, extra_prompt="", profile="standard"):
    if profile not in CUSTOMER_PROFILES:
        raise ValueError("Unknown customer profile: " + profile)
    additions = [text for text in (IDENTITY_STYLE_INSTRUCTION, extra_prompt, CUSTOMER_PROFILES[profile]) if text]
    return base_prompt + ("\n\n" + "\n\n".join(additions) if additions else "")


def lookup_recovery_instruction(scenario):
    """Give the actor concrete typo/correction values from its own known facts only."""
    instructions = scenario.get("instructions", {}) if isinstance(scenario, dict) else {}
    if not isinstance(instructions, dict):
        return "No structured known identity is available. Do not invent a typo or identifier."
    known = instructions.get("known_info") or ""
    unknown = instructions.get("unknown_info") or ""
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", known)
    if "email" in unknown.lower():
        emails = []
    zips = re.findall(r"(?<![A-Za-z0-9])(\d{5})(?![A-Za-z0-9])", known)
    if emails:
        correct = emails[0]
        local, domain = correct.split("@", 1)
        field, typo = "email", local + "x@" + domain
    elif zips:
        correct = zips[0]
        field, typo = "ZIP", correct[:-1] + str((int(correct[-1]) + 1) % 10)
    else:
        return "No known email or ZIP is supplied. Create no fictional identity value; follow the task and clarify honestly."
    return (
        f"CONCRETE ONE-TYPO SCENARIO: Your real {field} remains {correct}. "
        f"The FIRST time you provide your {field}, say {typo} instead, exactly once. "
        f"After the agent reports lookup failure or asks you to check it, acknowledge the typo "
        f"and give the exact real value {correct}. Never use {typo} again after correction. "
        "This is the sole intentional mistake; do not alter your name, goals, items, addresses, or order IDs."
    )
