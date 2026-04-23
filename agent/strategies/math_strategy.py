"""
Math strategy — for AIME-style competition math problems.

Techniques used:
  - Step-Back Prompting (Technique 3): identify principles first
  - Self-Refine (Technique 6): check and correct if answer looks wrong
  - Answer Extraction (Technique 7): robust number extraction

Call budget: 2-3 calls max
"""

from __future__ import annotations

import re

from agent.llm import reset_call_count
from agent.techniques.step_back import step_back
from agent.techniques.cot import chain_of_thought
from agent.techniques.self_refine import self_refine
from agent.extractor import extract_answer, extract_number


def _looks_wrong(answer: str) -> bool:
    """Heuristic: does this answer look like it needs refinement?"""
    if not answer:
        return True
    # Non-numeric for a math problem
    if not re.search(r"\d", answer):
        return True
    # Too long (should be a number, not a sentence)
    if len(answer) > 50:
        return True
    return False


def math_strategy(question: str) -> str:
    """
    Math strategy:
    1. Step-back CoT (identify principles, then solve)
    2. Extract answer
    3. If answer looks wrong → self-refine
    4. Extract again
    """
    reset_call_count()

    # Call 1: Step-back prompting
    raw = step_back(question)
    answer = extract_answer(raw, "math")

    # Call 2 (conditional): Self-refine if answer looks wrong
    if _looks_wrong(answer):
        raw2 = self_refine(question, answer or raw[:200], "math")
        answer2 = extract_answer(raw2, "math")
        if answer2 and not _looks_wrong(answer2):
            return answer2
        # If refine also failed, try plain CoT as last resort
        if _looks_wrong(answer2):
            raw3 = chain_of_thought(question, "math")
            answer3 = extract_answer(raw3, "math")
            if answer3:
                return answer3

    return answer or ""
