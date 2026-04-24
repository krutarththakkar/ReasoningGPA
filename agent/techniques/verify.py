"""
Verification — ask the model if its answer actually addresses the question.
Returns a bool so the caller can decide whether to trust it or reflect.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM = (
    "You are a strict verifier. "
    "Check whether the answer satisfies the question. "
    "Reply with exactly CORRECT or INCORRECT."
)


def verify(question: str, answer: str) -> bool:
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Does this answer correctly and completely answer the question? "
        "Reply with CORRECT or INCORRECT."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=50)
    return raw.strip().upper().startswith("CORRECT")
