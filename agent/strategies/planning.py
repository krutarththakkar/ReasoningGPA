"""
Planning strategy — solve PDDL-like action planning problems.
Expected answers are multi-line action lists like:
    (feast b d)
    (succumb b)
We prompt for the plan only, then filter out any prose.
"""
from __future__ import annotations

import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a careful planner. "
    "Given a planning problem, output ONLY the plan as a sequence of actions. "
    "Each action on its own line, in the form (action arg1 arg2 ...). "
    "Use lowercase and only action names from the problem. "
    "No numbering, no explanation, no markdown fences."
)


def _final_statement_only(question: str) -> str:
    """Keep the rules and the final unsolved planning statement."""
    if "[STATEMENT]" not in question:
        return question

    preamble, *statements = question.split("[STATEMENT]")
    final_statement = statements[-1].split("[PLAN]")[0].rstrip()
    return f"{preamble.rstrip()}\n\n[STATEMENT]{final_statement}\n\n[PLAN]"


def planning_strategy(question: str) -> str:
    reset_call_count()
    question = _final_statement_only(question)
    prompt = (
        f"{question}\n\n"
        "Return ONLY the plan, one action per line, each like (action arg1 arg2 ...).\n"
        "Use exact action names from the problem; do not invent wrapper actions like use.\n"
        "Do not include words like object, from, or another inside action lines.\n"
        "No numbering, no explanation, no markdown."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=1000)
    return _extract_plan(raw)


def _natural_action(line: str) -> str | None:
    line = line.strip().lower().rstrip(".")
    patterns = [
        (r"^attack object ([\w\-]+)$", "(attack {0})"),
        (r"^succumb object ([\w\-]+)$", "(succumb {0})"),
        (r"^feast object ([\w\-]+) from (?:another )?object ([\w\-]+)$", "(feast {0} {1})"),
        (r"^overcome object ([\w\-]+) from (?:another )?object ([\w\-]+)$", "(overcome {0} {1})"),
        (r"^pick up the ([\w\-]+) block$", "(pick-up {0})"),
        (r"^put down the ([\w\-]+) block$", "(put-down {0})"),
        (r"^unstack the ([\w\-]+) block from on top of the ([\w\-]+) block$", "(unstack {0} {1})"),
        (r"^stack the ([\w\-]+) block on top of the ([\w\-]+) block$", "(stack {0} {1})"),
    ]
    for pat, template in patterns:
        match = re.match(pat, line)
        if match:
            return template.format(*match.groups())
    return None


def _extract_plan(raw: str) -> str:
    """Pull out just the parenthesised action lines from the model's response."""
    if not raw:
        return ""
    s = raw.rstrip()

    # unwrap a markdown fence if the model used one
    fence = re.search(r"```\w*\s*\n(.*?)\n```", s, re.DOTALL)
    if fence:
        s = fence.group(1)

    # scan the whole text and keep anything that matches the (action args) shape
    # action/arg chars: lowercase letters, digits, hyphens, underscores
    pat = re.compile(r"\([a-zA-Z][\w\-]*(?:\s+[\w\-]+)*\)")
    actions = [
        action
        for m in pat.finditer(s)
        if (action := _clean_action(m.group(0))) is not None
    ]
    if not actions:
        actions = [
            action
            for line in s.splitlines()
            if (action := _natural_action(line)) is not None
        ]
        if not actions:
            return ""

    # expected answers end with a trailing newline — mimic that for exact match
    return "\n".join(actions) + "\n"


def _clean_action(action: str) -> str | None:
    parts = action.strip("()").lower().split()
    if not parts or parts[0] == "use":
        return None
    parts = [p for p in parts if p not in {"object", "from", "another"}]
    if len(parts) < 2:
        return None
    return "(" + " ".join(parts) + ")"
