"""
Commonsense strategy.

Plain CoT to get an answer, then verify it. If verification says it's
fine we return it, otherwise we reflect and try again.
"""

from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.cot import chain_of_thought
from agent.techniques.verify import verify
from agent.techniques.reflection import reflect
from agent.extractor import extract_answer


def commonsense_strategy(question: str) -> str:
    reset_call_count()

    raw = chain_of_thought(question, "commonsense")
    answer = extract_answer(raw, "commonsense")
    if not answer:
        return ""

    if verify(question, answer):
        return answer

    # verifier wasn't happy — take a second pass
    reflected = reflect(question, answer, "commonsense")
    return reflected or answer
