"""
Word problem strategy.

We run decomposition and a plain CoT, then compare. If both got the same
number we're good. If they disagree we ask the model to reflect on the
decomposition answer since that one breaks the problem down more carefully.
"""

from __future__ import annotations
import re

from agent.llm import reset_call_count
from agent.techniques.decompose import decompose
from agent.techniques.cot import chain_of_thought
from agent.techniques.reflection import reflect
from agent.extractor import pull_final_answer, last_numeric_token


def _is_multilingual(question: str) -> bool:
    non_ascii = sum(1 for c in question if ord(c) > 127)
    return non_ascii > 5


def word_problem_strategy(question: str) -> str:
    reset_call_count()

    q = question
    if _is_multilingual(question):
        q = question + "\n\n(Note: Understand the question in its original language and compute the numeric answer.)"

    raw_d = decompose(q)
    decomp_answer = pull_final_answer(raw_d, "word_problem")

    raw_c = chain_of_thought(q, "word_problem")
    cot_answer = pull_final_answer(raw_c, "word_problem")

    d_num = last_numeric_token(decomp_answer)
    c_num = last_numeric_token(cot_answer)
    if d_num and c_num and d_num == c_num:
        return d_num

    # they disagree — decomp is usually more careful so we reflect on that
    if decomp_answer:
        reflected = reflect(q, decomp_answer, "word_problem")
        if reflected and re.search(r"\d", reflected):
            return reflected
        return decomp_answer

    return cot_answer or ""
