"""
Technique 4: Least-to-Most Decomposition
Break word problems into sequential sub-problems.
Handles multi-step arithmetic reliably.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM = (
    "You are a careful problem solver. "
    "Break the problem into smaller steps and solve each one. "
    "Show your work for each step. "
    "State 'Final answer: <number>' at the very end."
)


def decompose(question: str) -> str:
    """
    Decomposition for word problems.
    Returns raw LLM response.
    """
    prompt = (
        "Break this problem into smaller steps:\n\n"
        f"Problem: {question}\n\n"
        "Step 1: What is the first quantity to find?\n"
        "Step 2: What is the next quantity?\n"
        "(Continue as needed)\n"
        "Final: Combine results to get the answer.\n\n"
        "Work through each step, then state 'Final answer: <number>'"
    )

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=600,
    )
