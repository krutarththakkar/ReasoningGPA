"""
MCQ strategy — multiple choice questions (science, general knowledge).

Techniques used:
  - Few-Shot Exemplars (Technique 5)
  - Chain-of-Thought (Technique 1)
  - Answer Extraction (Technique 7)

Call budget: 1 call
"""

from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.few_shot import few_shot
from agent.extractor import extract_answer


def mcq_strategy(question: str) -> str:
    """
    MCQ strategy:
    Few-shot CoT → extract letter answer.
    """
    reset_call_count()

    raw = few_shot(question, "science_mcq")
    answer = extract_answer(raw, "science_mcq")

    # make sure we return just the letter if possible
    import re
    if answer and re.match(r"^[A-D]$", answer.strip()):
        return answer.strip()

    # Try to extract letter from longer answer
    m = re.search(r"\b([A-D])\b", answer or raw)
    if m:
        return m.group(1).upper()

    return answer or ""
