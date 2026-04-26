"""
Coding strategy — generate a Python function body for the given task.
Expected answers are literal indented function bodies, no def line or imports.
"""
from __future__ import annotations

import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "This is only for coding. Make sure your answer is in python only. "
    "Use proper formatting, do not explain your code. "
    "Use proper reasoning to validate your answer and make sure you aren't running the code."
)


def _has_starter_signature(question: str) -> bool:
    return bool(re.search(r"self-contained code starting|def\s+task_func\s*\(", question, re.IGNORECASE))


def coding_strategy(question: str) -> str:
    reset_call_count()
    has_starter = _has_starter_signature(question)
    answer_shape = (
        "Return only the indented function body. Do not include the def line or imports."
        if has_starter
        else "Return a complete Python function definition. Include the def line."
    )
    prompt = (
        f"{question}\n\n"
        "This is only for coding. Make sure your answer is in python only. "
        "Use proper formatting, do not explain your code. "
        "Use proper reasoning to validate your answer and make sure you aren't running the code. "
        f"{answer_shape}"
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=1500)
    initial_code = _extract_code(raw, keep_signature=not has_starter)

    # Code Review Subagent
    reviewer_system = (
        "You are an expert Code Review Subagent. Your job is to review the code written by a junior developer "
        "to ensure it STRICTLY matches the requirements of the prompt.\n"
        "1. Did they add unnecessary comments, prints, or styling/font changes? If so, remove them.\n"
        "2. Did they hardcode file names or values that weren't explicitly requested? If so, fix them to use the provided constants or parameters.\n"
        "3. Did they use non-recursive APIs (like os.listdir) when recursive search (os.walk or rglob) was needed? If so, upgrade it.\n"
        "4. Does the code perfectly match the requested behavior and error messages? If the prompt specifies a ValueError, use that exact logic.\n"
        f"Output ONLY the corrected Python code inside a ```python block. {answer_shape} Do not explain your changes."
    )
    
    reviewer_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"JUNIOR DEVELOPER'S CODE:\n{initial_code}\n\n"
        "Review the code above. If it contains any of the mistakes listed in your system prompt, fix them. "
        f"Return ONLY the corrected Python code inside a ```python block. {answer_shape}"
    )

    reviewed_raw = call_llm(reviewer_prompt, system=reviewer_system, temperature=0.0, max_tokens=1500)
    return _extract_code(reviewed_raw, keep_signature=not has_starter)


def _extract_code(raw: str, keep_signature: bool = False) -> str:
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

    # drop a copied starter block if the model ignored instructions
    lines = s.split("\n")
    def_line = next(
        (idx for idx, line in enumerate(lines) if line.lstrip().startswith("def ")),
        None,
    )
    if def_line is not None and keep_signature:
        lines = lines[def_line:]
    elif def_line is not None:
        lines = lines[def_line + 1:]
    elif not keep_signature:
        lines = [line for line in lines if not line.lstrip().startswith(("import ", "from "))]

    first_code = next((line for line in lines if line.strip()), "")
    if first_code and not keep_signature and not first_code.startswith((" ", "\t")):
        lines = [("    " + line if line.strip() else line) for line in lines]

    return "\n".join(lines).rstrip()
