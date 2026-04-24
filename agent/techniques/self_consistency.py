"""
Self-Consistency — ask the model the same question N times with some
randomness, then pick whichever answer came up most. Works well on math
and logic where one bad step can derail an otherwise correct approach.
"""

from __future__ import annotations

import re
from collections import Counter

from agent.llm import call_llm
from agent.extractor import extract_answer

_SYSTEM = (
    "You are a careful problem solver. "
    "Think step by step. "
    "At the very end, write your final answer on its own line "
    "prefixed with exactly: 'Final answer:'"
)

_DOMAIN_HINTS = {
    "math": (
        "Solve this math problem. Show all your work. "
        "Compute the final numeric value. "
        "State 'Final answer: <number>' at the end."
    ),
    "word_problem": (
        "Solve this word problem step by step. "
        "State 'Final answer: <number>' at the end."
    ),
    "logic": (
        "Trace through the logic carefully, tracking all state changes. "
        "State 'Final answer: <answer>' at the end."
    ),
}


def _normalize(answer: str, domain: str) -> str:
    """Strip formatting so "42", "42.", and " 42 " all count as the same vote."""
    if not answer:
        return ""
    s = answer.strip().rstrip(".,;:!?").lower()

    if domain in ("math", "word_problem"):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if m:
            return m.group(0)

    if domain in ("logic", "science_mcq", "commonsense"):
        m = re.match(r"^\(?([a-d])\)?$", s)
        if m:
            return m.group(1).upper()

    return s


def self_consistency(question: str, domain: str, n: int = 3) -> str:
    """Run n samples and return whichever answer won the vote."""
    hint = _DOMAIN_HINTS.get(
        domain, "Solve step by step. State the final answer clearly."
    )
    prompt = f"{hint}\n\nQuestion: {question}"

    votes: list[str] = []
    original_by_key: dict[str, str] = {}  # so we return the real answer, not the normalized key

    for _ in range(n):
        raw = call_llm(
            prompt,
            system=_SYSTEM,
            temperature=0.7,
            max_tokens=800,
        )
        if not raw:
            continue
        ans = extract_answer(raw, domain)
        if not ans:
            continue
        key = _normalize(ans, domain)
        if not key:
            continue
        votes.append(key)
        original_by_key.setdefault(key, ans)

    if not votes:
        return ""

    winner_key = Counter(votes).most_common(1)[0][0]
    return original_by_key.get(winner_key, winner_key)
