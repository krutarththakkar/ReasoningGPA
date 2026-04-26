"""
Grader - determines if a prediction is correct.

Priority:
  1. Exact match after normalization (0 extra LLM calls)
  2. Numeric extraction match (0 extra LLM calls)
  3. LLM-as-judge (1 extra LLM call, only when needed)
"""

from __future__ import annotations

import re
from agent.extractor import normalize_for_grading, last_numeric_token


def mark_prediction(
    question: str,
    prediction: str,
    expected: str,
    use_llm_judge: bool = True,
    domain: str | None = None,
) -> bool:
    """
    Grade a prediction against the expected answer.
    Returns True if correct.
    """
    if not prediction:
        return False

    # Coding path - skips trivia normalizations that destroy Python syntax
    if domain == "coding":
        if prediction.strip() == expected.strip():
            return True
        if use_llm_judge:
            from agent.techniques.self_eval import self_evaluate
            return self_evaluate(question, prediction, expected, domain="coding")
        return False

    # Strict numeric path - used for math and any expected answer that is just
    # a number. Skips the LLM judge (which hallucinates on long AIME questions)
    # and skips substring match (which wrongly says "2" in "112" -> True).
    expected_is_numeric = bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", str(expected or "")))
    if domain == "math" or expected_is_numeric:
        pn = last_numeric_token(prediction)
        en = last_numeric_token(expected)
        return pn is not None and en is not None and pn == en

    # Normalize once
    pred_norm = normalize_for_grading(prediction)
    exp_norm  = normalize_for_grading(expected)

    # 1. Exact match after normalization
    if pred_norm == exp_norm:
        return True

    # 1.5 Handle logical equivalence
    if pred_norm in ("true", "yes") and exp_norm in ("true", "yes"):
        return True
    if pred_norm in ("false", "no") and exp_norm in ("false", "no"):
        return True

    # 2. Numeric match
    pred_num = last_numeric_token(prediction)
    exp_num  = last_numeric_token(expected)
    if pred_num is not None and exp_num is not None and pred_num == exp_num:
        return True

    # 3. Substring match (prediction contains expected or vice versa)
    # require the shorter string to be ≥3 chars to prevent single-char false positives
    if exp_norm and len(exp_norm) >= 3 and exp_norm in pred_norm:
        return True
    if pred_norm and len(pred_norm) >= 3 and pred_norm in exp_norm:
        return True

    # 4. LLM-as-judge (flexible matching)
    if use_llm_judge:
        from agent.techniques.self_eval import self_evaluate
        return self_evaluate(question, prediction, expected, domain=domain or "")

    return False
