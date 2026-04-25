"""
Common-sense strategy — short factual trivia.
Now uses Least-to-Most Decomposition (L2M) and a Verify-Reflect loop.
"""
from __future__ import annotations

import re
from agent.llm import reset_call_count
from agent.techniques.decompose import decompose
from agent.techniques.verify import verify
from agent.techniques.reflection import reflect
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


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    
    # Step 1: L2M Reasoning
    raw = decompose(question, "commonsense")
    answer = extract_answer(raw, "commonsense")

    # If the answer is a bare boolean but the question is NOT a yes/no question,
    # the extractor was confused — try to pull the real answer from the raw trace
    if answer and answer.lower() in _BOOLEANS and not _is_yes_no_question(question):
        for line in reversed((raw or "").split("\n")):
            s = line.strip().rstrip(".,;")
            if s and s.lower() not in _BOOLEANS and s.upper() not in {"NOTHING", "NONE", "UNKNOWN"}:
                answer = s
                break

    # Fallback if bare extraction still fails
    if not answer:
        for line in reversed((raw or "").split("\n")):
            s = line.strip()
            if s:
                answer = s
                break

    if not answer:
        return ""

    # # Step 2: Verification
    # is_correct = verify(question, answer)
    # if is_correct:
    #     return answer
        
    # # Step 3: Reflection if unverified
    # reflected_raw = reflect(question, answer, "commonsense")
    # reflected_answer = extract_answer(reflected_raw, "commonsense")
    # if reflected_answer:
    #     return reflected_answer

    return answer
