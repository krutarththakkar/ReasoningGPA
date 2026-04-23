"""
Technique 3: Step-Back Prompting
For hard math: identify relevant principles first, then solve.
Reduces "wrong method" errors on AIME-style problems.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM = (
    "You are an expert mathematician. "
    "First identify the key mathematical principles, theorems, and formulas needed. "
    "Then apply them carefully to solve the problem. "
    "Compute the final numeric value — do not leave it as an expression. "
    "State 'Final answer: <number>' at the very end."
)


def step_back(question: str) -> str:
    """
    Step-back prompting for hard math.
    Returns raw LLM response.
    """
    prompt = (
        "Step 1 — Identify relevant concepts:\n"
        "What mathematical concepts, theorems, or formulas are needed for this problem? "
        "List them briefly.\n\n"
        "Step 2 — Solve:\n"
        "Using those concepts, solve the problem step by step. "
        "Show all calculations.\n\n"
        "Step 3 — Final answer:\n"
        "State the final numeric answer as 'Final answer: <number>'\n\n"
        f"Problem: {question}"
    )

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=1200,
    )
