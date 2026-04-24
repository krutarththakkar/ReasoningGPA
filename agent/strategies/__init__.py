"""
Strategy registry — maps domain names to strategy functions.
Each strategy takes a question string and returns a clean answer string.
"""

from agent.strategies.math_strategy import math_strategy
from agent.strategies.word_problem import word_problem_strategy
from agent.strategies.reading_comp import reading_comp_strategy
from agent.strategies.mcq import mcq_strategy
from agent.strategies.logic import logic_strategy
from agent.strategies.true_false import true_false_strategy
from agent.strategies.commonsense import commonsense_strategy
from agent.strategies.coding import coding_strategy
from agent.strategies.planning import planning_strategy
from agent.strategies.future_prediction import future_prediction_strategy

_REGISTRY = {
    "math":                   math_strategy,
    "word_problem":           word_problem_strategy,
    "reading_comprehension":  reading_comp_strategy,
    "science_mcq":            mcq_strategy,
    "logic":                  logic_strategy,
    "true_false":             true_false_strategy,
    "commonsense":            commonsense_strategy,
    # Real dev-data labels — "common_sense" is the same as "commonsense".
    # Coding / planning / future_prediction get dedicated strategies in later
    # patches; for now they fall back to commonsense so they at least produce
    # something rather than crashing.
    "common_sense":           commonsense_strategy,
    "coding":                 coding_strategy,
    "planning":               planning_strategy,
    "future_prediction":      future_prediction_strategy,
}


def get_strategy(domain: str):
    """Return the strategy function for a given domain."""
    return _REGISTRY.get(domain, commonsense_strategy)
