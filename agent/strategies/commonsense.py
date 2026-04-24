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
    "You answer factual trivia. First, think step by step to arrive at the correct answer. "
    "At the very end, state 'Final answer: <answer>' where <answer> is just a name, place, date, or brief phrase."
)


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    prompt = (
        f"{question}\n\n"
        "Think step by step and then provide the final answer starting with 'Final answer:'."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=300)

    answer = extract_answer(raw, "commonsense")
    if answer:
        return answer

    # fallback: first non-empty line if the extractor couldn't find anything
    for line in (raw or "").split("\n"):
        s = line.strip()
        if s:
            return s
    return ""
