"""
Commonsense strategy — general knowledge, plausibility, similarity.

Techniques used:
  - Chain-of-Thought T1
  - Answer Extraction T7

Call budget: 1 call
"""

from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.cot import chain_of_thought
from agent.extractor import extract_answer


def commonsense_strategy(question: str) -> str:
    """
    Commonsense strategy:
    CoT → extract answer.
    """
    reset_call_count()

    raw = chain_of_thought(question, "commonsense")
    return extract_answer(raw, "commonsense")
