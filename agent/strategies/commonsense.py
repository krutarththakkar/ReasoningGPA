"""
Common-sense strategy — short factual trivia.
Pure CoT approach, dynamically handling boolean formatting.
"""
from __future__ import annotations

import re
from agent.llm import reset_call_count
from agent.techniques.cot import chain_of_thought
from agent.extractor import extract_answer

# Yes/No confirmation questions start with these words
_YES_NO_PATTERN = re.compile(
    r"^\s*(is|are|were|was|can|do|does|have|has|will|would|could|should|did"
    r"|are both|were both|is it|are they|was it)\b",
    re.IGNORECASE,
)

_BOOLEANS = {"true", "false", "yes", "no"}


def _is_yes_no_question(q: str) -> bool:
    """Return True if the question expects a yes/no/true/false answer."""
    return bool(_YES_NO_PATTERN.match(q.strip()))


def _is_boolean_answer(s: str) -> bool:
    """Return True if a string is or contains only a bare boolean."""
    cleaned = s.lower().strip()
    if cleaned in _BOOLEANS:
        return True
    for b in _BOOLEANS:
        if re.fullmatch(rf"(?:final\s+)?answer\s*[:\-]?\s*{b}", cleaned):
            return True
    return False


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    is_yn = _is_yes_no_question(question)

    # Pass in `is_yn` dynamically. This is critical to avoid False Positives!
    raw = chain_of_thought(question, "commonsense")
    answer = extract_answer(raw, "commonsense")

    # Boolean rejection guard
    if answer and _is_boolean_answer(answer) and not is_yn:
        answer = ""
        for line in reversed((raw or "").split("\n")):
            s = line.strip().rstrip(".,;")
            if s and not _is_boolean_answer(s) and s.upper() not in {"NOTHING", "NONE", "UNKNOWN"}:
                answer = s
                break

    if not answer:
        for line in reversed((raw or "").split("\n")):
            s = line.strip()
            if s and (is_yn or not _is_boolean_answer(s)):
                answer = s
                break

    return answer or ""
