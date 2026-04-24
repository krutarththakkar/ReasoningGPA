"""
Coding strategy — generate a Python function body for the given task.
Expected answers are literal indented function bodies, no def line or imports.
"""
from __future__ import annotations

import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a careful Python programmer. "
    "Given a task description, first think step-by-step about the algorithm, edge cases, and logic. "
    "After your reasoning, output ONLY the indented Python function body inside a ```python code block. "
    "No function signature, no imports. Keep the solution short and close to the requested behavior. "
    "Do not modernize or improve the task beyond what is asked. "
    "Every line in the code block must be indented with at least 4 spaces."
)


def coding_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"{question}\n\n"
        "First, explain your approach step-by-step.\n"
        "Then, return ONLY the indented Python function body that solves this task inside a ```python block.\n"
        "Use the imports, constants, and parameters from the provided starter code. "
        "Match the examples and requested behavior literally. "
        "Do not replace named APIs with equivalent alternatives. "
        "Do not synthesize fallback data or optional branches unless requested. "
        "Do not add extra validation, retries, pagination, logging, printing, "
        "directory creation, comments, or error handling unless the task asks for it. "
        "Do not add None checks or argument guards unless they are requested. "
        "Do not include the def line or imports inside the code block."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=1500)
    return _extract_code(raw)


def _extract_code(raw: str) -> str:
    """Pull just the function body out of the model's response."""
    if not raw:
        return ""
    # rstrip only — leading spaces matter (they're the body's indentation)
    s = raw.rstrip()

    # strip markdown fences if the model used them anyway
    fence = re.search(r"```(?:python)?\s*\n(.*?)\n```", s, re.DOTALL)
    if fence:
        s = fence.group(1)

    # drop a "Final answer:" prefix line — [^\n]* so we only eat that one line
    # and don't swallow the leading spaces of the code below
    s = re.sub(r"^Final answer\s*:[^\n]*\n", "", s, flags=re.IGNORECASE)
    s = s.lstrip("\n\r")

    # drop a leading "def ..." signature if the model ignored instructions
    lines = s.split("\n")
    if lines and lines[0].lstrip().startswith("def "):
        lines = lines[1:]

    return "\n".join(lines).rstrip()
