"""
Answer extractor — post-processes raw LLM output into a clean final answer.
Zero LLM calls. Pure string processing.

Handles all output formats the model might produce:
  - "Final answer: 112"
  - "\\boxed{112}"
  - "**112**"
  - "Therefore, the answer is 112."
  - "m + n = 112"
  - "The answer is B."
  - "(C) blue ball"
  - "Yes, because..."
"""

from __future__ import annotations

import re


def extract_answer(raw: str, domain: str) -> str:
    """
    Extract a clean final answer from raw LLM output.
    Returns the best candidate answer string.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Priority 1: Explicit "Final answer:" marker
    m = re.search(
        r"(?:final answer|the answer is|answer is)\s*[:\-]?\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip(".,;")
        if candidate:
            return _domain_clean(candidate, domain)

    # Priority 2: LaTeX boxed
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # Priority 3: Bold markdown **answer**
    m = re.search(r"\*\*([^*]+)\*\*", text)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) < 100:
            return _domain_clean(candidate, domain)

    # Priority 4: "Therefore/Thus/So, X"
    m = re.search(
        r"(?:therefore|thus|so|hence),?\s+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip(".,;")
        if candidate and len(candidate) < 100:
            return _domain_clean(candidate, domain)

    # Priority 5: "m + n = X" or "a + b + c = X" patterns (math)
    if domain == "math":
        m = re.search(r"[a-z]\s*\+\s*[a-z](?:\s*\+\s*[a-z])?\s*=\s*(\d+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # "= X" at end of line
        m = re.search(r"=\s*(\d+(?:\.\d+)?)\s*\.?\s*$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()

    # Priority 6: Domain-specific extraction
    result = _domain_specific(text, domain)
    if result:
        return result

    # Priority 7: Last short line heuristic
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        last = lines[-1].rstrip(".,;")
        if len(last) < 150:
            return _domain_clean(last, domain)

    # Fallback: truncate raw
    return text[:200]


def _domain_specific(text: str, domain: str) -> str:
    """Domain-specific extraction patterns."""

    if domain in ("science_mcq", "logic", "commonsense"):
        # Extract single letter answer: "B", "(B)", "B.", "B)"
        m = re.search(r"(?:^|\s|\()([A-D])(?:\)|\.|\s|$)", text)
        if m:
            return m.group(1).upper()

    if domain == "true_false":
        lower = text.lower()
        # Check first 100 chars for yes/no
        first = lower[:100]
        if re.search(r"\byes\b", first):
            return "Yes"
        if re.search(r"\bno\b", first):
            return "No"

    if domain == "math":
        # Last number in the text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

    if domain == "word_problem":
        # Last number in the text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

    if domain == "reading_comprehension":
        # Try to find a short answer span
        # Look for "is X" or "was X" patterns
        m = re.search(r"(?:is|was|were|are)\s+([A-Z][^.]{0,80}?)(?:\.|$)", text)
        if m:
            return m.group(1).strip()

    return ""


def _domain_clean(candidate: str, domain: str) -> str:
    """Clean up a candidate answer based on domain expectations."""

    if domain in ("science_mcq", "logic"):
        # If it starts with a letter + period/paren, extract just the letter
        m = re.match(r"^([A-D])[\.\)\s]", candidate)
        if m:
            return m.group(1).upper()
        # If it's just a letter
        if re.match(r"^[A-D]$", candidate.strip()):
            return candidate.strip().upper()

    if domain == "true_false":
        lower = candidate.lower().strip()
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"

    if domain == "math":
        # Strip trailing punctuation and whitespace
        candidate = candidate.rstrip(".,;:")
        # If it's a clean number, return it
        if re.match(r"^-?\d+(?:\.\d+)?$", candidate.strip()):
            return candidate.strip()

    return candidate.strip()


def normalize_for_grading(answer: str) -> str:
    """
    Normalize an answer for comparison with expected output.
    Mirrors the tutorial's normalize_text function.
    """
    s = (answer or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_number(s: str) -> str | None:
    """Extract the first number from a string."""
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else None
