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


def self_evaluate(question: str, prediction: str, expected: str = "", domain: str = "") -> bool:
    """
    Ask the model if the prediction is correct.
    Returns True if correct, False otherwise (including when the judge's reply is ambiguous).
    """
    if domain == "coding" and expected:
        prompt = (
            f"QUESTION: {question[:800]}\n\n"
            f"EXPECTED CODE:\n{expected}\n\n"
            f"PREDICTED CODE:\n{prediction}\n\n"
            "You are an expert Python grader. Analyze the EXPECTED code and the PREDICTED code. "
            "If the PREDICTED code perfectly solves the stated QUESTION, return True. "
            "The EXPECTED code is just one possible solution. Do not penalize the PREDICTED code if it "
            "omits unstated error handling, uses different variable names, uses non-recursive functions when recursion isn't explicitly asked for, "
            "or misses extra features (like specific error messages or constraints) that are present in the EXPECTED code but NOT explicitly required by the QUESTION. "
            "Return True if the logic fundamentally satisfies the core QUESTION. "
            "Reply with exactly: True or False"
        )
    elif expected:
        prompt = (
            f"QUESTION: {question[:400]}\n\n"
            f"PREDICTION: {prediction}\n\n"
            f"EXPECTED: {expected}\n\n"
            "Is the prediction correct for this question? "
            "A prediction is correct if it identifies the same person, place, or fact as the expected answer, "
            "even if it uses a different name format (e.g. 'Richard Nixon' matches 'President Richard Nixon'). "
            "A prediction is WRONG if it names a different entity, gives a different number, or is a boolean (True/False) "
            "when the expected answer is a name or place. "
            "If there are options provided, the prediction MUST match one of the options exactly. "
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
