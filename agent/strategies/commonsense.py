"""
Common-sense strategy — short factual trivia.
Pure CoT approach, dynamically handling boolean formatting.
"""
from __future__ import annotations

import re
from agent.llm import reset_call_count
from agent.techniques.cot import chain_of_thought
from agent.techniques.debate import debate
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


def _should_debate(question: str) -> bool:
    q = question.lower()
    return bool(
        re.search(
            r"options?:|\([a-d]\)|similar to|most likely|\bbest\b|\bwhich\b|find a|choose",
            q,
        )
    )


def _extract_with_guard(raw: str, question: str, is_yn: bool) -> str:
    answer = extract_answer(raw, "commonsense")

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
            s = re.sub(r"^\d+[\)\.]\s*", "", s).strip().rstrip(".,;")
            if s and len(s) > 2 and (is_yn or not _is_boolean_answer(s)): #no numbered lists
                answer = s
                break

    return answer or ""


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    is_yn = _is_yes_no_question(question)

    if not is_yn and _should_debate(question):
        debated = debate(question, "commonsense")
        if debated and not _is_boolean_answer(debated):
            return debated

    raw = chain_of_thought(question, "commonsense")
    return _extract_with_guard(raw, question, is_yn)
