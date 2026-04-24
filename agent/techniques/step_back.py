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
    "Write plain text only. Do NOT use markdown headers, bold, or asterisks. "
    "The very last line of your response must be exactly: Final answer: <number>"
)


def step_back(question: str) -> str:
    """
    Step-back prompting for hard math.
    Returns raw LLM response.
    """
    prompt = (
        f"Problem: {question}\n\n"
        "Solve this in plain prose. Do not use headers, bullet points, or bold text.\n\n"
        "First, briefly list the mathematical concepts, theorems, or formulas relevant here. "
        "Then work through the solution, showing all calculations. "
        "Compute the final numeric value — do not leave it as an expression.\n\n"
        "End with this on its own line as the very last line:\n"
        "Final answer: <number>"
    )

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=2000,
    )
