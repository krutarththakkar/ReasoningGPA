"""
Common-sense strategy — short factual trivia. One call, tight prompt.

Questions are things like "Which magazine was started first?" or "What city
hosts Oberoi's head office?". The model knows these; asking for reasoning +
verify + reflect just wastes calls and occasionally overrides a correct answer.
"""
from __future__ import annotations

from agent.llm import call_llm, reset_call_count
from agent.extractor import extract_answer

_SYSTEM = (
    "You are careful with factual trivia and entity-linking questions. "
    "Use the clues in the question, avoid famous-but-wrong guesses, and "
    "end with one line exactly like: Final answer: <short answer>"
)


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"{question}\n\n"
        "Identify the exact entity, date, place, title, or phrase being asked for. "
        "If the question compares two named choices, compare those choices and pick the one that fits. "
        "If it gives a chain of clues, follow every clue before answering. "
        "Keep any reasoning brief, then end with 'Final answer: <answer>'."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=220)

    answer = extract_answer(raw, "commonsense")
    if answer:
        return answer

    # fallback: first non-empty line if the extractor couldn't find anything
    for line in (raw or "").split("\n"):
        s = line.strip()
        if s:
            return s
    return ""
