"""
Future-prediction strategy — the prompt already tells the model to end with
\\boxed{X}. We extract that X and wrap it in the Python-like list format the
expected answers use: ['Yes'], [265.0], ['A', 'B', 'C'].
"""
from __future__ import annotations

import re

import ast
import statistics
from collections import Counter

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are an elite Superforecaster. You must make a precise prediction about a future event based strictly on your internal knowledge. "
    "You cannot search the internet. You must extrapolate from historical data. "
    "Never refuse to predict. Always end with \\boxed{YOUR_PREDICTION}."
)


def future_prediction_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"QUESTION:\n{question}\n\n"
        "To make the most accurate prediction, use the following structured forecasting process:\n"
        "1. HISTORICAL BASERATE: Recall the exact historical numbers, prices, or winners for this entity up to your knowledge cutoff.\n"
        "2. TREND EXTRAPOLATION: Identify the rate of change or seasonal effects leading up to the target date.\n"
        "3. PROBABILITY DISTRIBUTION: If this is a number, establish a realistic Lower Bound and Upper Bound. If this is multiple choice, assign a percentage probability to each option.\n"
        "4. FINAL ESTIMATE: Calculate your median/expected value.\n\n"
        "You MUST output your final answer inside \\boxed{}. "
        "If it is a number, output just the number. If it is multiple choice, output the exact letter(s) or option text required."
    )
    predictions = []
    for _ in range(5):
        raw = call_llm(prompt, system=_SYSTEM, temperature=0.7, max_tokens=1500)
        p = _format_prediction(raw)
        if p:
            predictions.append(p)
            
    if not predictions:
        raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=1500)
        return _format_prediction(raw)
        
    parsed_preds = []
    for p in predictions:
        try:
            val = ast.literal_eval(p)
            if isinstance(val, list) and val:
                parsed_preds.append(val)
        except Exception:
            pass
            
    if not parsed_preds:
        return predictions[0]
        
    is_numeric = all(len(v) == 1 and isinstance(v[0], (int, float)) for v in parsed_preds)
    if is_numeric:
        nums = [v[0] for v in parsed_preds]
        median = statistics.median(nums)
        return f"[{median}]"
        
    tuples = [tuple(v) for v in parsed_preds]
    most_common = Counter(tuples).most_common(1)[0][0]
    return "[" + ", ".join(repr(item) for item in most_common) + "]"

def _format_prediction(raw: str) -> str:
    """Pull content from the LAST \\boxed{...} and wrap it as a Python list string."""
    if not raw:
        return ""
    matches = re.findall(r"\\boxed\{([^{}]*)\}", raw)
    if not matches:
        for line in reversed(raw.split("\n")): 
            output = line.strip()
            if output:
                return f"['{output}']" #last nonempty line
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
