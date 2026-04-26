"""
Technique: Iterative Self-Ask Agent
Multi-hop commonsense trivia reasoning.

Pipeline:
  1. Decompose  — identify ≤3 atomic sub-questions (1 call)
  2. Resolve    — answer each hop, feeding prior answers as context (≤3 calls)
  3. Synthesize — derive the final answer from the full context (1 call)

5 LLM calls maximum, well within the 20-call budget.
"""

from __future__ import annotations

import re

from agent.llm import call_llm

_SYSTEM_DECOMPOSE = (
    "You are a careful analyst. "
    "Given a question, identify the atomic sub-questions that must be answered IN ORDER to reach the final answer. "
    "Output ONLY a numbered list of sub-questions, one per line. "
    "No preamble, no explanation. Maximum 3 sub-questions."
)

_SYSTEM_RESOLVE = (
    "You are a precise factual question answerer. "
    "You will be given any facts already established and a single sub-question. "
    "Answer the sub-question concisely and accurately in one sentence. "
    "Output only the answer, nothing else."
)

_SYSTEM_SYNTHESIZE = (
    "You are a concise answer extractor. "
    "Use the provided established facts to answer the original question. "
    "Give only the shortest possible direct answer — a name, place, year, or brief phrase only. "
    "If the question offers two named choices, you MUST select exactly ONE of the two given options. "
    "Do NOT write 'Neither', 'both', or any explanation. "
    "Do NOT answer with True or False unless the question explicitly asks for a yes/no confirmation. "
    "State 'Final answer: <answer>' at the very end — where <answer> is the name, word, or brief phrase only."
)


def decompose_hops(question: str) -> list[str]:
    """
    Ask the model to list the atomic sub-questions needed to answer the question.
    Returns a list of up to 3 sub-question strings.
    """
    prompt = (
        f"Question: {question}\n\n"
        "List the atomic sub-questions that need to be answered IN ORDER to resolve this. "
        "Output a numbered list only. Maximum 3 sub-questions."
    )
    raw = call_llm(prompt, system=_SYSTEM_DECOMPOSE, temperature=0.0, max_tokens=300)
    hops: list[str] = []
    for line in (raw or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1." or "1)"
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            hops.append(cleaned)
    return hops[:3]


def resolve_hop(sub_question: str, context: str) -> str:
    """
    Answer a single sub-question, injecting all prior answers as grounded context.
    Returns the raw answer string.
    """
    context_block = f"Established facts so far:\n{context}\n\n" if context.strip() else ""
    prompt = (
        f"{context_block}"
        f"Sub-question: {sub_question}\n"
        "Answer concisely in one sentence."
    )
    raw = call_llm(prompt, system=_SYSTEM_RESOLVE, temperature=0.0, max_tokens=150)
    return (raw or "").strip()


def self_ask(question: str) -> str:
    """
    Full iterative Self-Ask pipeline.
    Returns raw LLM synthesis text; pass through pull_final_answer() for cleaning.
    """
    # Phase 1: Decompose into sub-questions
    hops = decompose_hops(question)
    if not hops:
        # Fallback: treat the whole question as one hop
        hops = [question]

    # Phase 2: Resolve each hop, accumulating context
    context_lines: list[str] = []
    for hop in hops:
        answer = resolve_hop(hop, "\n".join(context_lines))
        if answer:
            context_lines.append(f"Q: {hop}\nA: {answer}")

    # Phase 3: Synthesize final answer from accumulated context
    context = "\n\n".join(context_lines)
    prompt = (
        f"Original question: {question}\n\n"
        f"Established facts obtained step by step:\n{context}\n\n"
        "Based only on these facts, give the single shortest direct answer to the original question. "
        "If the question names two options, you MUST pick exactly one of the two options. Do not say 'neither' or 'both' or 'it depends'. State '<answer>' at the end "
        "where <answer> is a name, word, or brief phrase only, NOT a sentence."
    )
    return call_llm(prompt, system=_SYSTEM_SYNTHESIZE, temperature=0.0, max_tokens=400)
