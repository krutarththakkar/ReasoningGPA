"""
Multi-agent debate.

Two independent samples solve the same question. If they disagree, a judge
compares both traces and returns the final answer.
"""
from __future__ import annotations

from agent.extractor import extract_answer, normalize_for_grading
from agent.llm import call_llm

_SOLVER_SYSTEM = (
    "You are an independent reasoner. Solve carefully and state "
    "'Final answer: <answer>' at the end."
)

_JUDGE_SYSTEM = (
    "You are a strict debate judge. Pick the answer with the soundest reasoning. "
    "If both are weak, solve it yourself. State 'Final answer: <answer>' at the end."
)

_HINTS = {
    "commonsense": (
        "Use commonsense and factual reasoning. For option questions, choose only "
        "from the provided options. Keep the final answer short."
    ),
    "logic": (
        "Track every state change carefully. For multiple choice, give the final "
        "letter or short option text."
    ),
}


def debate(question: str, domain: str) -> str:
    hint = _HINTS.get(domain, "Reason carefully and give a short final answer.")
    prompt = f"{hint}\n\nQuestion: {question}\n\nThink step by step."

    raw_a = call_llm(prompt, system=_SOLVER_SYSTEM, temperature=0.7, max_tokens=600)
    raw_b = call_llm(prompt, system=_SOLVER_SYSTEM, temperature=0.7, max_tokens=600)

    ans_a = extract_answer(raw_a, domain)
    ans_b = extract_answer(raw_b, domain)
    if ans_a and not ans_b:
        return ans_a
    if ans_b and not ans_a:
        return ans_b
    if not ans_a and not ans_b:
        return ""
    if normalize_for_grading(ans_a) == normalize_for_grading(ans_b):
        return ans_a

    judge_prompt = (
        f"Question:\n{question}\n\n"
        f"Agent A reasoning:\n{raw_a}\n\nAgent A answer: {ans_a}\n\n"
        f"Agent B reasoning:\n{raw_b}\n\nAgent B answer: {ans_b}\n\n"
        "Which agent's reasoning is flawless? If neither is flawless, correct them. "
        "Output the final answer only at the end as 'Final answer: <answer>'."
    )
    judged = call_llm(judge_prompt, system=_JUDGE_SYSTEM, temperature=0.0, max_tokens=500)
    return extract_answer(judged, domain) or ans_a or ans_b
