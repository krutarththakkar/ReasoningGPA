"""Exact arithmetic search for small expression prompts."""
from __future__ import annotations

import re
from fractions import Fraction

from agent.llm import reset_call_count


def expression_puzzle_strategy(question: str) -> str:
    reset_call_count()
    values = _read_card_values(question)
    target = _read_goal_value(question)
    expression = _build_expression([(Fraction(n), str(n)) for n in values], target)
    return f"Solution: {expression}" if expression else "Solution: "


def _read_card_values(question: str) -> list[int]:
    match = re.search(r"Numbers?\s*:\s*([0-9,\s.-]+)", question, re.IGNORECASE)
    source = match.group(1) if match else question
    return [int(token) for token in re.findall(r"-?\d+", source)[:4]]


def _read_goal_value(question: str) -> Fraction:
    patterns = [
        r"equals?\s+(-?\d+)",
        r"make\s+(-?\d+)",
        r"form\s+(?:an\s+)?expression\s+that\s+equals\s+(-?\d+)",
        r"(\d+)-Game",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return Fraction(int(match.group(1)), 1)
    return Fraction(0, 1)


def _build_expression(cards: list[tuple[Fraction, str]], target: Fraction) -> str:
    if len(cards) == 1:
        return cards[0][1] if cards[0][0] == target else ""

    for left_idx in range(len(cards)):
        for right_idx in range(left_idx + 1, len(cards)):
            remaining = [
                item for idx, item in enumerate(cards)
                if idx not in (left_idx, right_idx)
            ]
            for merged in _pairwise_results(cards[left_idx], cards[right_idx]):
                answer = _build_expression(remaining + [merged], target)
                if answer:
                    return answer
    return ""


def _pairwise_results(first, second):
    first_value, first_text = first
    second_value, second_text = second

    yield first_value + second_value, f"({first_text} + {second_text})"
    yield first_value - second_value, f"({first_text} - {second_text})"
    yield second_value - first_value, f"({second_text} - {first_text})"
    yield first_value * second_value, f"({first_text} * {second_text})"

    if second_value:
        yield first_value / second_value, f"({first_text} / {second_text})"
    if first_value:
        yield second_value / first_value, f"({second_text} / {first_text})"
