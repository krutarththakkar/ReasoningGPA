"""
Domain router — classifies questions into domains using heuristics only.
0 LLM calls. >> fast and deterministic.

Domains: math, word_problem, reading_comprehension, science_mcq,
         logic, true_false, commonsense
"""

from __future__ import annotations

import re


# Science keywords that distinguish science_mcq from logic mcq
_SCIENCE_KEYWORDS = {
    "energy", "cell", "organism", "planet", "chemical", "force", "wave",
    "atom", "gene", "ecosystem", "photosynthesis", "erosion", "condensation",
    "evaporation", "precipitation", "circulatory", "digestive", "respiratory",
    "excretory", "phase", "solvent", "reaction", "element", "compound",
    "gravity", "magnetic", "electric", "thermal", "kinetic", "potential",
    "nucleus", "membrane", "chromosome", "evolution", "habitat", "species",
    "density", "pressure", "temperature", "volume", "mass", "weight",
    "light", "sound", "heat", "electricity", "magnetism", "radiation",
    "fossil", "mineral", "rock", "soil", "water cycle", "food chain",
    "photon", "electron", "proton", "neutron", "molecule", "ion",
}

# Math LaTeX patterns — must NOT match plain dollar amounts like "$4"
_LATEX_PATTERNS = [
    r"\$[^$\d][^$]*\$",    # inline math $...$ (not starting with digit — avoids $4)
    r"\$\\",               # math starting with backslash like $\frac
    r"\\frac\{",           # fractions
    r"\\sqrt\{",           # square roots
    r"\\triangle",         # triangles
    r"\\angle",            # angles
    r"\\binom\{",          # binomial coefficients
    r"\\log",              # logarithms
    r"\\sin|\\cos|\\tan",  # trig
    r"\\le|\\ge|\\neq",    # inequalities
    r"\\cdot|\\times",     # multiplication
    r"\\sum|\\prod",       # summation/product
    r"\\int",              # integrals
    r"\\lfloor|\\lceil",   # floor/ceiling
    r"\\mathcal|\\mathbb", # math fonts
    r"\^\{",               # superscripts like x^{2}
    r"_\{",                # subscripts like a_{n}
]

# Math keyword patterns (non-Latex)
_MATH_KEYWORDS = [
    r"\bprime\b", r"\binteger\b", r"\bdivisible\b", r"\bremainder\b",
    r"\bprobability\b", r"\btriangle\b", r"\bcircle\b", r"\bpolygon\b",
    r"\bquadrilateral\b", r"\bsequence\b", r"\barithmetic\b",
    r"\bgeometric\b", r"\bmodulo\b", r"\bfactorial\b", r"\bpermutation\b",
    r"\bcombination\b", r"\bmatrix\b", r"\bdeterminant\b", r"\beigenvalue\b",
    r"\bcongruent\b", r"\bsimilar\b", r"\bparallel\b", r"\bperpendicular\b",
    r"\bmedian\b", r"\baltitude\b", r"\bbisector\b", r"\bincircle\b",
    r"\bcircumscribed\b", r"\binscribed\b", r"\btangent\b", r"\bchord\b",
    r"\bparabola\b", r"\bellipse\b", r"\bhyperbola\b",
    r"\bfind\s+\w+\s+if\b", r"\bcompute\b", r"\bevaluate\b",
    # "how many X" where X is a math object (integers, primes, etc.)
    r"\bhow many\s+(even|odd|prime|positive|negative|distinct|different)\b",
    r"\bhow many\s+\w+\s+(between|less than|greater than|from \d)\b",
    r"\bfind the (number|sum|product|area|volume|length|distance|angle|probability)\b",
]

_COMPETITION_MATH_PATTERNS = [
    r"\bAIME\b",
    r"\bhow many ways\b",
    r"\bpositive integers?\b",
    r"\binteger-sided\b",
    r"\bnoncongruent\b",
    r"\bcommon perimeter\b",
    r"\bminimum possible value\b",
    r"\bmaximum possible value\b",
    r"\bsets? of two\b",
    r"\bdisjoint subsets?\b",
    r"\bfolded in half repeatedly\b",
    r"\bunit squares?\b.*\bfolded\b",
    r"\bratio of the lengths?\b",
    r"\bnumber of (sets|ways|ordered pairs|solutions|integers)\b",
    r"\b(largest|smallest|least|greatest) number\b",
]

# Word problem patterns
_WORD_PROBLEM_PATTERNS = [
    r"\$\s*\d",                          # $ amounts
    r"\d+\s*(dollars?|cents?|euros?)",   # currency
    r"\d+\s*(percent|%)",                # percentages
    r"\d+\s*(kg|pounds?|miles?|km)",     # units
    r"\d+\s*(hours?|days?|weeks?|years?|minutes?)",  # time
    r"\bhow many more\b",
    r"\bhow much\b",
    r"\btotal\b.*\d",
    r"\bcalculate\b",
    r"\bearn(ed|s)?\b",
    r"\bbought?\b",
    r"\bsell?s?\b",
    r"\bpay(s|ed)?\b",
    r"\bcost(s|ed)?\b",
    r"\bweigh(s|ed)?\b",
    r"\bsave(d|s)?\b",
    r"\bspend(s)?\b",
    r"\bspent\b",
    r"\bprofit\b",
    r"\bloss\b",
    r"\bdiscount\b",
    r"\bcommission\b",
    r"\ballowance\b",
]

# Logic puzzle patterns — state tracking, swapping, ordering
# Must NOT match movie/similarity questions that happen to have (A)(B)(C) options
_LOGIC_PATTERNS = [
    r"Options:\s*\n?\s*\(A\).+\n.+\(B\)",   # multi-line options (logic puzzles)
    r"At the end.*(?:Bob|Alice|Claire)\s+has",
    r"swap(ped|s)?\s+balls?",
    r"\btrade(d|s)?\s+balls?",
    r"pairs?\s+of\s+players?\s+trade",
    r"\bposition\b.*\brace\b",
    r"First,\s+\w+\s+and\s+\w+\s+swap",
    # Race position puzzle
    r"pass\s+the\s+person\s+in\s+\w+\s+place",
    r"what\s+position\s+are\s+you",
]

# Commonsense signals -> questions about similarity, recommendations, plausibility
_COMMONSENSE_SIGNALS = [
    r"\bsimilar\s+to\b",
    r"\bfind\s+a\s+movie\b",
    r"\bmost\s+likely\b",
    r"\bbest\s+describes\b",
    r"\bplausible\b",
]

# True/False patterns —> must be specific, not catch mcq questions
_TRUE_FALSE_PATTERNS = [
    r"^Facts?:",                          # starts with "Facts:"
    r"\nFacts?:",                         # "Facts:" on its own line
    r"\s+Facts?:",                        # "Facts:" after whitespace
    r"Is\s+.+\s+true\?",
    r"Is\s+.+\s+false\?",
    r"Does\s+.+\s+have\s+a\s+monopoly",  # specific monopoly pattern
    r"Is\s+the\s+following\s+sentence\s+plausible",
    r"Is\s+the\s+\w+.{0,30}known\s+for\s+being",  # "Is X known for being Y?"
]

# Reading comprehension patterns
_RC_PATTERNS = [
    r"Context:",
    r"Passage:",
    r"\[DOC\]",
    r"\[PAR\]",
    r"\[TLE\]",
]

# Dev-label domains (coding / planning / future_prediction). Very specific
# scaffolding phrases so false positives on other domains are 0.
_CODING_MARKERS = [
    r"self-contained code starting",
    r"The function should output",
    r"def\s+task_func",
]
_PLANNING_MARKERS = [
    r"Here are the actions I can do",
    r"Here are the actions that can be performed",
    r"I am playing with a set of objects",
    r"I have to plan",
]
_FUTURE_PREDICTION_MARKERS = [
    r"agent that can predict future events",
    r"\\boxed\{YOUR_PREDICTION\}",
]


def _has_math_dollar_span(question: str) -> bool:
    """True for LaTeX-ish $...$ spans, false for plain money amounts."""
    for match in re.finditer(r"\$([^$]{1,80})\$", question):
        span = match.group(1).strip()
        if not span:
            continue
        if re.search(r"[\\{}_^=<>+\-*/]", span):
            return True
        if re.fullmatch(r"\.?\d+(?:\.\d+)?", span):
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", span):
            return True
    return False


def _has_competition_math_signal(question: str) -> bool:
    return any(
        re.search(pat, question, re.IGNORECASE)
        for pat in _COMPETITION_MATH_PATTERNS
    )


def _looks_multilingual_word_problem(question: str) -> bool:
    non_ascii = sum(1 for c in question if ord(c) > 127)
    if non_ascii < 4 or not re.search(r"\d", question):
        return False

    numbers = re.findall(r"\d+(?:[.,]\d+)?", question)
    has_question = any(mark in question for mark in "?؟？")
    has_arithmetic_hint = bool(
        re.search(
            r"[%$€£¥+\-*/×÷]|倍|半|分の|多少|几|幾|何|combien|cu[aá]nt|"
            r"calcula|คำนวณ|กี่|ทั้งหมด|ครึ่ง|หนึ่งใน|สองใน|कितन|ఎంత|কত",
            question,
            re.IGNORECASE,
        )
    )

    return len(numbers) >= 2 or has_question or has_arithmetic_hint


def detect_domain(question: str) -> str:
    """
    Classify question into one of the supported domains using heuristics.
    Returns: coding | planning | future_prediction | math | word_problem |
             reading_comprehension | science_mcq | logic | true_false | commonsense
    """
    q = question.strip()
    q_lower = q.lower()

    # Check the three dev-data domains first — their markers are very specific
    for pat in _FUTURE_PREDICTION_MARKERS:
        if re.search(pat, q, re.IGNORECASE):
            return "future_prediction"
    for pat in _CODING_MARKERS:
        if re.search(pat, q, re.IGNORECASE):
            return "coding"
    for pat in _PLANNING_MARKERS:
        if re.search(pat, q, re.IGNORECASE):
            return "planning"

    # True/False (check early, has explicit "Facts:" marker)
    for pat in _TRUE_FALSE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "true_false"

    # Reading comprehension (long context with explicit markers)
    for pat in _RC_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "reading_comprehension"

    # Long question with "Context" in it
    if len(q) > 800 and re.search(r"\bcontext\b", q_lower):
        return "reading_comprehension"

    # Logic (state tracking, swapping — check before MCQ) 
    for pat in _LOGIC_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "logic"

    # Science MCQ (A/B/C/D options + science keywords)
    has_abcd = bool(re.search(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", q))
    if has_abcd:
        words = set(q_lower.split())
        if words & _SCIENCE_KEYWORDS:
            return "science_mcq"
        # MCQ without science keywords → still MCQ
        if re.search(r"\bA\.\s.+\bB\.\s.+\bC\.\s.+\bD\.\s", q, re.DOTALL):
            return "science_mcq"

    # Commonsense signals -> check AFTER MCQ to avoid misclassification
    for pat in _COMMONSENSE_SIGNALS:
        if re.search(pat, q, re.IGNORECASE):
            return "commonsense"

    # Word problem — check BEFORE math to avoid dollar sign confusion
    # Strong word problem signals (2+ hits or 1 very strong signal)
    if _has_math_dollar_span(q) or _has_competition_math_signal(q):
        return "math"

    if _looks_multilingual_word_problem(q):
        return "word_problem"

    word_problem_hits = sum(
        1 for pat in _WORD_PROBLEM_PATTERNS
        if re.search(pat, q, re.IGNORECASE)
    )
    # Plain dollar amounts like "$4" or "$1000" are word problems, not math
    has_plain_dollar = bool(re.search(r"\$\s*\d+(?:\.\d+)?(?!\s*[a-zA-Z\\])", q))
    if has_plain_dollar and word_problem_hits >= 1:
        return "word_problem"
    if word_problem_hits >= 2:
        return "word_problem"

    # Math (LaTeX or math keywords)
    for pat in _LATEX_PATTERNS:
        if re.search(pat, q):
            return "math"

    for pat in _MATH_KEYWORDS:
        if re.search(pat, q_lower):
            return "math"

    # Single strong word problem signal (after math check)
    if word_problem_hits >= 1:
        return "word_problem"

    # Commonsense (default)
    return "commonsense"
