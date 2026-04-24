"""
Math strategy.

First does a step-back pass (think about the principles, then solve),
then runs self-consistency on the side (3 random samples, majority vote).
If both agree on the same number we're confident — return early.
If they disagree, try to self-refine and, if that still looks off, fall
back to a plain CoT as a last resort.
"""

from __future__ import annotations

import re

from agent.llm import reset_call_count
from agent.techniques.step_back import step_back
from agent.techniques.cot import chain_of_thought
from agent.techniques.self_refine import self_refine
from agent.techniques.self_consistency import self_consistency
from agent.extractor import extract_answer, extract_number


def _looks_wrong(answer: str) -> bool:
    """True if the answer is empty, has no digits, or is suspiciously long."""
    if not answer:
        return True
    if not re.search(r"\d", answer):
        return True
    if len(answer) > 50:
        return True
    return False


def math_strategy(question: str) -> str:
    reset_call_count()

    raw_sb = step_back(question)
    sb_answer = extract_answer(raw_sb, "math")

    sc_answer = self_consistency(question, "math", n=3)

    # If both methods landed on the same number, we're confident
    sb_num = extract_number(sb_answer)
    sc_num = extract_number(sc_answer)
    if sb_num and sc_num and sb_num == sc_num:
        return sb_num

    candidate = sb_answer if not _looks_wrong(sb_answer) else sc_answer

    if _looks_wrong(candidate):
        seed = candidate or sb_answer or sc_answer or raw_sb[:200]
        raw_ref = self_refine(question, seed, "math")
        ref_answer = extract_answer(raw_ref, "math")
        if ref_answer and not _looks_wrong(ref_answer):
            return ref_answer

        # nothing worked — try a plain CoT as last resort
        raw_cot = chain_of_thought(question, "math")
        cot_answer = extract_answer(raw_cot, "math")
        if cot_answer:
            return cot_answer

    return candidate or sc_answer or sb_answer or ""
