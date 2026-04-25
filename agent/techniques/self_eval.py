"""
Technique 8: LLM-as-Judge (Self-Evaluation)
Use the model to verify whether an answer is correct.
Used in eval pipeline and optionally in the agent for math verification.
"""

from __future__ import annotations

import re

from agent.llm import call_llm

_SYSTEM = (
    "You are a strict grader. "
    "Reply with exactly True or False. "
    "No punctuation. No explanation."
)


def self_evaluate(question: str, prediction: str, expected: str = "") -> bool:
    """
    Ask the model if the prediction is correct.
    Returns True if correct, False otherwise (including when the judge's reply is ambiguous).
    """
    if expected:
        prompt = (
            f"QUESTION: {question[:400]}\n\n"
            f"PREDICTION: {prediction}\n\n"
            f"EXPECTED: {expected}\n\n"
            "Is the prediction semantically correct for this question? "
            "Important rules:\n"
            "- Partial names are correct if they clearly identify the same person (e.g. 'Richard Nixon' matches 'President Richard Nixon').\n"
            "- Alternate but equivalent phrasings are correct (e.g. 'ethyl alcohol' matches 'alcohol').\n"
            "- A city name without a country/state qualifier is correct if it unambiguously identifies the same place.\n"
            "- Do NOT penalize for omitting titles like 'President', 'Dr.', 'Sir' unless the question specifically asks for the title.\n"
            "Reply with exactly: True or False"
        )
    else:
        prompt = (
            f"QUESTION: {question[:400]}\n\n"
            f"ANSWER: {prediction}\n\n"
            "Is this answer correct? "
            "Reply with exactly: True or False"
        )

    reply = call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=10,
    ).strip().lower()

    # strict word-boundary match — "False" wins over "True" so "not true" etc. aren't false positives
    if re.search(r"\bfalse\b", reply):
        return False
    if re.search(r"\btrue\b", reply):
        return True

    # judge didn't give a clear answer — be conservative, don't inflate scores
    return False
