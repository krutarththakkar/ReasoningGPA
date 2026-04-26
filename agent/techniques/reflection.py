"""
Reflection — give the model an answer and ask if it's actually right.
Unlike self_refine (which is math-only and checks calculations), this
works on any domain and just asks "is this correct, and if not, fix it."
"""

from __future__ import annotations

from agent.llm import call_llm
from agent.extractor import pull_final_answer

_SYSTEM = (
    "You are a careful reviewer. "
    "Check whether the given answer is correct for the question. "
    "If it is wrong or uncertain, solve the problem and give the correct answer. "
    "State 'Final answer: <answer>' at the end."
)


def reflect(question: str, initial_answer: str, domain: str) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"Proposed answer: {initial_answer}\n\n"
        "Is this answer correct? If not, what is the correct answer? "
        "Think carefully and state 'Final answer: <answer>' at the end."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=500)
    return pull_final_answer(raw, domain)
