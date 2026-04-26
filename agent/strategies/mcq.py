"""
Science MCQ strategy.

Few-shot tends to get a clean letter fast. If not, try a CoT pass. If
that still didn't give a letter, do a reflection — sometimes the model
gives the right answer but buries the letter in a paragraph and the
reflection pass gets it to say just the letter.
"""

from __future__ import annotations
import re

from agent.llm import reset_call_count
from agent.techniques.few_shot import few_shot
from agent.techniques.cot import chain_of_thought
from agent.techniques.reflection import reflect
from agent.extractor import pull_final_answer


def _is_letter(s: str) -> bool:
    return bool(s and re.match(r"^[A-D]$", s.strip()))


def mcq_strategy(question: str) -> str:
    reset_call_count()

    raw_fs = few_shot(question, "science_mcq")
    fs_answer = pull_final_answer(raw_fs, "science_mcq")
    if _is_letter(fs_answer):
        return fs_answer.strip().upper()

    raw_cot = chain_of_thought(question, "science_mcq")
    cot_answer = pull_final_answer(raw_cot, "science_mcq")
    if _is_letter(cot_answer):
        return cot_answer.strip().upper()

    candidate = fs_answer or cot_answer
    if candidate:
        reflected = reflect(question, candidate, "science_mcq")
        if _is_letter(reflected):
            return reflected.strip().upper()

    # still no letter — just scan everything for any A-D
    for text in (fs_answer, cot_answer, raw_fs, raw_cot):
        m = re.search(r"\b([A-D])\b", text or "")
        if m:
            return m.group(1).upper()

    return fs_answer or cot_answer or ""
