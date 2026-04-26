"""
Logic strategy — ball swapping, ordering, state-tracking puzzles.

Self-consistency is the main move here: these puzzles have a definite
right answer, so if 2 out of 3 samples agree we can trust it. Falls back
to a single careful CoT if all 3 samples somehow fail.
"""

from __future__ import annotations

from agent.llm import call_llm, reset_call_count
from agent.techniques.debate import debate
from agent.techniques.self_consistency import self_consistency
from agent.extractor import pull_final_answer

_SYSTEM = (
    "You are a careful logical reasoner. "
    "Track all state changes explicitly. "
    "State 'Final answer: <answer>' at the end."
)


def logic_strategy(question: str) -> str:
    reset_call_count()

    debated = debate(question, "logic")
    if debated:
        return debated

    sc_answer = self_consistency(question, "logic", n=3)
    if sc_answer:
        return sc_answer

    prompt = (
        "Solve this logic problem by tracking all state changes step by step.\n\n"
        f"Problem: {question}\n\n"
        "For each step, explicitly state what changed and the current state.\n"
        "Then state 'Final answer: <answer>'"
    )

    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=500)
    return pull_final_answer(raw, "logic")
