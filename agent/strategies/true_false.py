"""
True/False strategy — evaluate a claim against provided facts.

Techniques used:
  - Few-Shot Exemplars (t5)
  - Answer Extraction (t7)

Call budget: 1 call
"""

from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.few_shot import few_shot
from agent.extractor import pull_final_answer


def true_false_strategy(question: str) -> str:
    """
    True/False strategy:
    Few-shot → extract Yes/No.
    """
    reset_call_count()

    raw = few_shot(question, "true_false")
    answer = pull_final_answer(raw, "true_false")

    # Normalize to Yes/No
    lower = (answer or "").lower().strip()
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"

    # Check raw response if extraction failed
    raw_lower = raw.lower()
    if raw_lower.startswith("yes") or "\nyes" in raw_lower[:100]:
        return "Yes"
    if raw_lower.startswith("no") or "\nno" in raw_lower[:100]:
        return "No"

    return answer or ""
