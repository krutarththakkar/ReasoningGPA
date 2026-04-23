"""
Technique 6: Self-Refine
After getting an initial answer, ask the model to check and correct it.
Only triggered for math when the initial answer looks wrong.
Costs 1 extra LLM call.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM = (
    "You are a careful reviewer and mathematician. "
    "Check the given answer for errors. "
    "If it is wrong, find the correct answer. "
    "Show your verification work. "
    "State 'Final answer: <number>' at the very end."
)


def self_refine(question: str, initial_answer: str, domain: str = "math") -> str:
    """
    Self-refine: review and correct an initial answer.
    Returns raw LLM response with corrected answer.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"A student answered: {initial_answer}\n\n"
        "Please check this answer carefully:\n"
        "1. Is the approach correct?\n"
        "2. Are the calculations correct?\n"
        "3. Is the final answer correct?\n\n"
        "If the answer is wrong, solve it correctly. "
        "State 'Final answer: <number>' at the end."
    )

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=1000,
    )
