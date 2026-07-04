"""Fairness guardrail.

Credit decisions must not use protected attributes. The guardrail
inspects incoming questions to the underwriting copilot and (a) refuses
to factor protected attributes into any assessment, (b) explains why.

This is deliberately a hard filter in front of the model, not a prompt
suggestion — prompts can be talked around, filters cannot.
"""

import re

PROTECTED = {
    "religion": ["hindu", "muslim", "christian", "sikh", "jain", "buddhist", "religion", "religious"],
    "caste/community": ["caste", "brahmin", "dalit", "obc", "sc/st", "scheduled caste", "scheduled tribe", "community certificate"],
    "gender": ["woman applicant", "female applicant", "male applicant", "because she is a woman", "because he is a man", "gender"],
    "marital status": ["unmarried", "divorced", "widow", "single mother", "marital status"],
    "region/ethnicity": ["north indian", "south indian", "ethnicity", "native place"],
}

_DECISION_WORDS = re.compile(
    r"\b(approve|decline|reject|deny|risky|risk|score|eligib|lend|sanction|trust)\w*", re.I
)


def check(question: str) -> dict | None:
    """Return a guardrail verdict if the question ties a protected
    attribute to a credit judgment; None if clean."""
    q = question.lower()
    hits = [
        group for group, words in PROTECTED.items()
        if any(w in q for w in words)
    ]
    if not hits:
        return None
    if not _DECISION_WORDS.search(q):
        # mentions an attribute but not in a decisioning context — note it, allow
        return {"level": "note", "groups": hits}
    return {"level": "block", "groups": hits}


def block_message(groups: list[str]) -> str:
    listed = ", ".join(groups)
    return (
        f"I can't factor {listed} into any credit assessment. "
        "Protected attributes are excluded from decisioning by design — "
        "both as regulatory requirement and because they add no signal a "
        "repayment-capacity model should use. I can assess this file on "
        "income stability, obligations, banking conduct, employment tenure, "
        "and collateral. Ask me about any of those."
    )
