"""
Technique 5: Few-Shot Exemplars
Include 1-2 solved examples in the prompt to establish expected format.
Zero extra LLM calls — examples are in the prompt itself.
"""

from __future__ import annotations

from agent.llm import call_llm

_SYSTEM = (
    "You are a careful problem solver. "
    "Follow the format shown in the examples. "
    "State 'Final answer: <answer>' at the end."
)

# Domain-specific examples
_EXAMPLES = {
    "science_mcq": (
        "Example:\n"
        "Question: Which process converts sunlight into chemical energy stored in food?\n"
        "A. respiration  B. photosynthesis  C. digestion  D. fermentation\n"
        "Reasoning: Photosynthesis uses sunlight to produce glucose.\n"
        "Final answer: B\n\n"
    ),
    "true_false": (
        "Example:\n"
        "Facts: Dogs are mammals. Mammals are warm-blooded.\n"
        "Question: Are dogs warm-blooded?\n"
        "Reasoning: Dogs are mammals, and mammals are warm-blooded, so yes.\n"
        "Final answer: Yes\n\n"
        "Example:\n"
        "Facts: The Eiffel Tower is in Paris. Paris is in France.\n"
        "Question: Is the Eiffel Tower in Germany?\n"
        "Reasoning: The Eiffel Tower is in Paris, which is in France, not Germany.\n"
        "Final answer: No\n\n"
    ),
    "reading_comprehension": (
        "Example:\n"
        "Context: The Eiffel Tower was built in 1889 in Paris, France.\n"
        "Question: When was the Eiffel Tower built?\n"
        "Final answer: 1889\n\n"
    ),
    "commonsense": (
        "Example:\n"
        "Question: Is it plausible that a basketball player scored 200 points in one game?\n"
        "Reasoning: The NBA record is 100 points. 200 is not plausible.\n"
        "Final answer: No\n\n"
    ),
    "logic": (
        "Example:\n"
        "Alice has a red ball, Bob has a blue ball. They swap. Who has the red ball?\n"
        "Reasoning: After the swap, Bob has the red ball.\n"
        "Final answer: Bob\n\n"
    ),
}


def few_shot(question: str, domain: str) -> str:
    """
    Few-shot prompting with domain-specific examples.
    Returns raw LLM response.
    """
    example = _EXAMPLES.get(domain, "")
    prompt = f"{example}Now solve:\nQuestion: {question}\n"

    return call_llm(
        prompt,
        system=_SYSTEM,
        temperature=0.0,
        max_tokens=400,
    )
