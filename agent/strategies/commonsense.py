"""
Common-sense strategy — short factual trivia.
Now uses Chain-of-Thought (CoT) to improve accuracy on multi-hop questions.
"""
from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.cot import chain_of_thought
from agent.extractor import extract_answer


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    raw = chain_of_thought(question, "commonsense")
    
    answer = extract_answer(raw, "commonsense")
    if answer:
        return answer

    # fallback: first non-empty line of the tail
    for line in reversed((raw or "").split("\n")):
        s = line.strip()
        if s:
            return s
    return ""
