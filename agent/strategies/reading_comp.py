"""
Reading comprehension strategy — answer questions from a provided context passage.

Techniques used:
  - Few-Shot Exemplars T5
  - Chain-of-Thought T5
  - Answer Extraction T7

Call budget: 1 call
"""

from __future__ import annotations
from agent.llm import call_llm, reset_call_count
from agent.extractor import pull_final_answer

_SYSTEM = (
    "You are a reading comprehension expert. "
    "Answer questions based only on the provided context. "
    "Give the shortest possible answer — a name, date, number, or brief phrase. "
    "Do not include surrounding context or explanation. "
    "State 'Final answer: <answer>' at the end."
)


def reading_comp_strategy(question: str) -> str:
    """
    Reading comprehension strategy:
    Single call with a tight extraction prompt.
    """
    reset_call_count()

    prompt = (
        "Read the following and answer the question with the shortest possible answer.\n"
        "Answer with just the key fact — a name, date, number, or short phrase.\n\n"
        f"{question}\n\n"
        "State 'Final answer: <answer>'"
    )

    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=200)
    return pull_final_answer(raw, "reading_comprehension")
