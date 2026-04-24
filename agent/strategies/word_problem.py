"""
Word problem strategy — arithmetic in natural language...

Techniques used:
  - Least-to-Most Decomposition T4
  - Chain-of-Thought fallback T1
  - Answer Extraction T7

Call budget: 1-2 calls
"""

from __future__ import annotations
import re

from agent.llm import reset_call_count
from agent.techniques.decompose import decompose
from agent.techniques.cot import chain_of_thought
from agent.extractor import extract_answer


def _is_multilingual(question: str) -> bool:
    """Detect non-ASCII characters suggesting non-English question."""
    non_ascii = sum(1 for c in question if ord(c) > 127)
    return non_ascii > 5


def word_problem_strategy(question: str) -> str:
    """
    Word problem strategy:
    1. Decomposition CoT
    2. Extract number
    3. If extraction fails → plain CoT fallback
    """
    reset_call_count()

    # For multilingual questions, add a language hint
    q = question
    if _is_multilingual(question):
        q = question + "\n\n(Note: Understand the question in its original language and compute the numeric answer.)"

    # Call 1: Decomposition
    raw = decompose(q)
    answer = extract_answer(raw, "word_problem")

    # If we got a number, we're done
    if answer and re.search(r"\d", answer):
        return answer

    # Call 2 (fallback): Plain CoT
    raw2 = chain_of_thought(q, "word_problem")
    answer2 = extract_answer(raw2, "word_problem")
    return answer2 or answer or ""
