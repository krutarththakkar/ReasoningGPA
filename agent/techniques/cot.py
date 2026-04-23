"""
Technique 1: Chain-of-Thought (CoT)
Ask the model to reason step by step before giving the final answer.
Applied to all domains as the base reasoning technique.
"""
from __future__ import annotations
from agent.llm import call_llm

_SYSTEM = (
    "You are a careful problem solver. "
    "Think step by step. "
    "At the very end, write your final answer on its own line "
    "prefixed with exactly: 'Final answer:'"
)

_DOMAIN_HINTS = {
    "math": (
        "Solve this math problem carefully. Show all steps. "
        "Compute the final numeric value — do not leave it as an expression. "
        "State 'Final answer: <number>' at the end."
    ),
    "word_problem": (
        "Solve this word problem step by step. "
        "Compute each quantity explicitly. "
        "State 'Final answer: <number>' at the end."
    ),
    "reading_comprehension": (
        "Read the context carefully. "
        "Answer based only on the information provided. "
        "Give the shortest possible answer that directly answers the question. "
        "State 'Final answer: <answer>' at the end."
    ),
    "science_mcq": (
        "Choose the best answer from the options. "
        "Explain your reasoning briefly, then state the letter. "
        "State 'Final answer: <letter>' at the end."
    ),
    "logic": (
        "Trace through the logic carefully, step by step. "
        "Keep track of all state changes. "
        "State 'Final answer: <answer>' at the end."
    ),
    "true_false": (
        "Evaluate the claim against the provided facts. "
        "State 'Final answer: Yes' or 'Final answer: No' at the end."
    ),
    "commonsense": (
        "Use common sense reasoning. "
        "State 'Final answer: <answer>' at the end."
    ),
}

# Token budgets per domain — math needs room to show work
_MAX_TOKENS = {
    "math": 1200,
    "word_problem": 600,
    "reading_comprehension": 400,
    "science_mcq": 300,
    "logic": 400,
    "true_false": 200,
    "commonsense": 300,
}


def chain_of_thought(question: str, domain: str) -> str:
    """
    Single CoT pass. Returns raw LLM response.
    Caller is responsible for extracting the final answer.
    """
    hint = _DOMAIN_HINTS.get(domain, "")
    max_tokens = _MAX_TOKENS.get(domain, 400)

    prompt = f"{hint}\n\nQuestion: {question}\n\nSolve step by step:"

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=max_tokens,
    )
