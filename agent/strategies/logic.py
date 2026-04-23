"""
Logic strategy — state tracking, ball swapping, ordering puzzles.

Techniques used:
  - Chain-of-Thought with explicit state tracking T1
  - Answer Extraction T7

Call budget: 1 call
"""

from __future__ import annotations

from agent.llm import call_llm, reset_call_count
from agent.extractor import extract_answer

_SYSTEM = (
    "You are a careful logical reasoner. "
    "Track all state changes explicitly. "
    "State 'Final answer: <answer>' at the end."
)


def logic_strategy(question: str) -> str:
    """
    Logic strategy:
    Step-by-step state tracking CoT.
    """
    reset_call_count()

    prompt = (
        "Solve this logic problem by tracking all state changes step by step.\n\n"
        f"Problem: {question}\n\n"
        "For each step, explicitly state what changed and the current state.\n"
        "Then state 'Final answer: <answer>'"
    )

    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=500)
    return extract_answer(raw, "logic")
