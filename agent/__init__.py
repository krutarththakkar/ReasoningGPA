"""
Entry point: agent_loop(question) -> str

Architecture:
  question -> router (domain detection) -> strategy -> technique(s) -> extractor -> answer
"""

from agent.router import detect_domain
from agent.strategies import get_strategy
from agent.llm import call_llm

def agent_loop(question: str) -> str:
    """
    Main entry point called by generate_answer_template.py.
    Routes question to the appropriate domain strategy and returns a clean answer string.
    """
    domain = detect_domain(question)
    strategy = get_strategy(domain)
    return strategy(question)
