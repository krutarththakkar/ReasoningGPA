from __future__ import annotations
import re

_SCIENCE_KEYWORDS = {
}

_LATEX_PATTERNS = [
]

_MATH_KEYWORDS = [
]

_WORD_PROBLEM_PATTERNS = [
]
_LOGIC_PATTERNS = [
]

_TRUE_FALSE_PATTERNS = [
]
_RC_PATTERNS = [
 ]


def detect_domain(question: str) -> str:
    q = question.strip()
    q_lower = q.lower()
    for pat in _TRUE_FALSE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "true_false"
    for pat in _RC_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "reading_comprehension"

    if len(q) > 800 and re.search(r"\bcontext\b", q_lower):
        return "reading_comprehension"
    for pat in _LOGIC_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "logic"

    has_abcd = bool(re.search(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", q))
    if has_abcd:
        words = set(q_lower.split())
        if words & _SCIENCE_KEYWORDS:
            return "science_mcq"
        if re.search(r"\bA\.\s.+\bB\.\s.+\bC\.\s.+\bD\.\s", q, re.DOTALL):
            return "science_mcq"

    for pat in _LATEX_PATTERNS:
        if re.search(pat, q):
            return "math"

    for pat in _MATH_KEYWORDS:
        if re.search(pat, q_lower):
            return "math"
    word_problem_hits = sum(
        1 for pat in _WORD_PROBLEM_PATTERNS
        if re.search(pat, q, re.IGNORECASE)
    )
    if word_problem_hits >= 2:
        return "word_problem"

    if re.search(r"\bhow many more weeks?\b|\bhow many more days?\b", q_lower):
        return "word_problem"

    return "commonsense"
