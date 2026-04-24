"""
Future-prediction strategy — the prompt already tells the model to end with
\\boxed{X}. We extract that X and wrap it in the Python-like list format the
expected answers use: ['Yes'], [265.0], ['A', 'B', 'C'].
"""
from __future__ import annotations

import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a careful predictor. Make a confident best-guess prediction. "
    "Follow any format instructions in the question, and always end your "
    "response with \\boxed{YOUR_PREDICTION}."
)


def future_prediction_strategy(question: str) -> str:
    reset_call_count()
    prompt = f"{question}\n\nReason briefly, then end with \\boxed{{YOUR_PREDICTION}}."
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=600)
    return _format_prediction(raw)


def _format_prediction(raw: str) -> str:
    """Pull content from the LAST \\boxed{...} and wrap it as a Python list string."""
    if not raw:
        return ""
    matches = re.findall(r"\\boxed\{([^{}]*)\}", raw)
    if not matches:
        return ""
    content = matches[-1].strip()
    if not content:
        return ""

    # already looks like a list? trust it
    if content.startswith("[") and content.endswith("]"):
        return content

    # number? expected format is always float-ish ("[265.0]" not "[265]")
    try:
        return f"[{float(content)}]"
    except ValueError:
        pass

    # otherwise split by comma and format as list of quoted strings
    parts = [p.strip().strip("'").strip('"') for p in content.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return "[" + ", ".join(f"'{p}'" for p in parts) + "]"
