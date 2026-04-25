"""
Common-sense strategy — short factual trivia.
Now uses Least-to-Most Decomposition (L2M) and a Verify-Reflect loop.
"""
from __future__ import annotations

from agent.llm import reset_call_count
from agent.techniques.decompose import decompose
from agent.techniques.verify import verify
from agent.techniques.reflection import reflect
from agent.extractor import extract_answer


def commonsense_strategy(question: str) -> str:
    reset_call_count()
    
    # Step 1: L2M Reasoning
    raw = decompose(question, "commonsense")
    answer = extract_answer(raw, "commonsense")
    
    # Fallback if bare extraction fails
    if not answer:
        for line in reversed((raw or "").split("\n")):
            s = line.strip()
            if s:
                answer = s
                break

    if not answer:
        return ""

    # Step 2: Verification
    is_correct = verify(question, answer)
    if is_correct:
        return answer
        
    # Step 3: Reflection if unverified
    reflected_raw = reflect(question, answer, "commonsense")
    reflected_answer = extract_answer(reflected_raw, "commonsense")
    if reflected_answer:
        return reflected_answer

    return answer
