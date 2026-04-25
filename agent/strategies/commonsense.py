"""
Common-sense strategy — short factual trivia.

Routing:
  - Yes/No questions  → single L2M decompose call (fast, 1 call)
  - Entity questions  → iterative Self-Ask agent (robust, ≤5 calls)
"""
from __future__ import annotations

import re
from agent.llm import reset_call_count
from agent.techniques.decompose import decompose
from agent.techniques.self_ask import self_ask
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


def _contains_only_boolean(s: str) -> bool:
    """Return True if a line is (or contains) only a bare boolean with no real content."""
    cleaned = s.lower().strip()
    # Exact match
    if cleaned in _BOOLEANS:
        return True
    # Compound patterns like "final answer: false", "answer: true", "the answer is no"
    for b in _BOOLEANS:
        if re.fullmatch(rf"(final\s+)?answer\s*[:\-]?\s*{b}", cleaned):
            return True
    return False


def _extract_with_fallback(raw: str, question: str) -> str:
    """Extract answer and apply boolean-rejection guard for entity questions."""
    answer = extract_answer(raw, "commonsense")

    # If it returned a bare boolean but this isn't a yes/no question,
    # walk backward through the raw trace and find the real answer
    if answer and _contains_only_boolean(answer) and not _is_yes_no_question(question):
        answer = ""
        for line in reversed((raw or "").split("\n")):
            s = line.strip().rstrip(".,;")
            if s and not _contains_only_boolean(s) and s.upper() not in {"NOTHING", "NONE", "UNKNOWN"}:
                answer = s
                break

    # General fallback: take the last non-empty, non-boolean line from raw
    if not answer:
        for line in reversed((raw or "").split("\n")):
            s = line.strip()
            if s and (not _is_yes_no_question(question) or not _contains_only_boolean(s)):
                answer = s
                break

    return answer


def commonsense_strategy(question: str) -> str:
    reset_call_count()

    if _is_yes_no_question(question):
        # Fast path: yes/no questions don't need multi-hop chain resolution
        raw = decompose(question, "commonsense")
    else:
        # Multi-hop path: iterative Self-Ask for entity/fact questions
        raw = self_ask(question)

    answer = _extract_with_fallback(raw, question)
    return answer or ""

