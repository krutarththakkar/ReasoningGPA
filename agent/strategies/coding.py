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
    "Write the simple reference-style solution, not a polished rewrite. "
    "Do not modernize, optimize, or improve the task beyond what is asked. "
    "Every line must be indented with at least 4 spaces."
)


def coding_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"{question}\n\n"
        "Return ONLY the indented Python function body that solves this task. "
        "The starter code is already provided by the grader; do not repeat imports or def lines. "
        "Use the imports, constants, and parameters from the provided starter code. "
        "Match the examples and requested behavior literally. "
        "If an imported module is clearly included for the task, use that module visibly "
        "instead of replacing it with a shortcut method. "
        "Preserve the random sequence shown by examples; seed exactly as requested and "
        "do not invent fixed seeds or conditional seed guards. "
        "Do not replace named APIs with equivalent alternatives. "
        "Do not synthesize fallback data or optional branches unless requested. "
        "If a parameter default is None, do not branch on it unless the prompt explicitly "
        "says callers may pass their own value. "
        "For running files and returning exit codes, prefer starting a process and waiting "
        "for its exit code over higher-level wrappers. "
        "For HTTP JSON tasks with the json module in the starter code, parse response.text "
        "with json instead of bypassing the module. "
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

    # drop a copied starter block if the model ignored instructions
    lines = s.split("\n")
    def_line = next(
        (idx for idx, line in enumerate(lines) if line.lstrip().startswith("def ")),
        None,
    )
    if def_line is not None:
        lines = lines[def_line + 1:]
    else:
        lines = [line for line in lines if not line.lstrip().startswith(("import ", "from "))]

    first_code = next((line for line in lines if line.strip()), "")
    if first_code and not first_code.startswith((" ", "\t")):
        lines = [("    " + line if line.strip() else line) for line in lines]

    return "\n".join(lines).rstrip()
