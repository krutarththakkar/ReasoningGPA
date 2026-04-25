"""
Technique 4: Least-to-Most Decomposition
Break problems into sequential sub-problems.
Handles multi-step arithmetic and multi-hop trivia reliably.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM_MATH = (
    "You are a careful problem solver. "
    "Break the problem into smaller steps and solve each one. "
    "Show your work for each step. "
    "State 'Final answer: <number>' at the very end."
)

_SYSTEM_COMMONSENSE = (
    "You are a careful problem solver testing facts step by step. "
    "Identify the exact entity, date, place, title, demographic, or phrase being asked for. "
    "If the answer is a person, provide their commonly spoken name with their title if applicable. "
    "If the question compares two named choices, compare those choices and pick the one that fits. "
    "If the question is a Yes/No or True/False question, your final answer MUST be exactly 'True' or 'False'. "
    "If it gives a chain of clues, follow every clue before answering. "
    "State 'Final answer: <answer>' at the very end."
)


def decompose(question: str, domain: str = "word_problem") -> str:
    """
    Decomposition for word problems and complex trivia.
    Returns raw LLM response.
    """
    if domain == "commonsense":
        prompt = (
            "Break this complex question into a sequence of simpler sub-questions to ensure you don't hallucinate.\n\n"
            f"Question: {question}\n\n"
            "Sub-question 1: What intermediate fact do you need to identify first?\n"
            "Answer 1: [Solve it]\n"
            "Sub-question 2: What is the next logical fact?\n"
            "Answer 2: [Solve it]\n"
            "Sub-question 3: What is the next logical fact, if needed?\n"
            "Answer 3: [Solve it]\n"
            "Final: Combine the sub-answers to resolve the original question.\n\n"
            "Work through each step carefully, then state 'Final answer: <answer>'"
        )
        sys_msg = _SYSTEM_COMMONSENSE
    else:
        prompt = (
            "Break this problem into smaller steps:\n\n"
            f"Problem: {question}\n\n"
            "Step 1: What is the first quantity to find?\n"
            "Step 2: What is the next quantity?\n"
            "Step 3: What is the next quantity, if needed?\n"
            "Final: Combine results to get the answer.\n\n"
            "Work through each step, then state 'Final answer: <number>'"
        )
        sys_msg = _SYSTEM_MATH

    return call_llm(
        prompt,
        system=sys_msg,
        temperature=0.0,
        max_tokens=600,
    )
