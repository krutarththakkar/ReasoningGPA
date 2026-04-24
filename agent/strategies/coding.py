"""
Coding strategy — generate a Python function body for the given task.
Expected answers are literal indented function bodies, no def line or imports.
"""
from __future__ import annotations

import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a careful Python programmer. "
    "Given a task description, output ONLY the function body. "
    "No function signature, no imports, no markdown code fences, no explanation. "
    "Every line must be indented with at least 4 spaces."
)


def coding_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"{question}\n\n"
        "Return ONLY the indented Python function body that solves this task. "
        "Do not include the def line, imports, or any explanation. "
        "Do not wrap the answer in markdown code fences."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=800)
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
