"""
Planning strategy — solve PDDL-like action planning problems.
Expected answers are multi-line action lists like:
    (feast b d)
    (succumb b)
We prompt for the plan only, then filter out any prose.
"""
from __future__ import annotations

import re
from collections import deque

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a careful planner. "
    "Think step-by-step. First, trace the initial state, the goal state, and keep track of the state of the world after each action. "
    "After your reasoning, output the final plan inside a ```plan code block. "
    "Inside the code block, put each action on its own line, in the form (action arg1 arg2 ...). "
    "Use lowercase and only action names from the problem."
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
    block_plan = _solve_blocks_problem(question)
    if block_plan:
        return block_plan
    attack_plan = _solve_attack_feast_problem(question)
    if attack_plan:
        return attack_plan
    is_logistics = "transporting crates" in question.lower()
    prompt_question = question if is_logistics else _final_statement_only(question)
    example_note = (
        "Use earlier [PLAN] blocks only as examples. Output only the final empty [PLAN].\n"
        if is_logistics else ""
    )
    prompt = (
        f"{prompt_question}\n\n"
        f"{example_note}"
        "Solve the planning problem step-by-step.\n"
        "At the end, output the plan inside a ```plan block, one action per line, each like (action arg1 arg2 ...).\n"
        "Use exact action names from the problem; do not invent wrapper actions like use.\n"
        "Do not include words like object, from, or another inside action lines."
    )
    raw = call_llm(prompt, system=_SYSTEM, temperature=0.0, max_tokens=1500)
    return _extract_plan(raw)


def _solve_blocks_problem(question: str) -> str:
    if "set of blocks" not in question.lower():
        return ""

    parsed = _parse_blocks_problem(question)
    if not parsed:
        return ""

    blocks, start_on, goals = parsed
    start = (tuple(start_on[b] for b in blocks), None)
    plan = _blocks_bfs(blocks, goals, start)
    if not plan:
        return ""
    return "\n".join(plan) + "\n"


def _parse_blocks_problem(question: str):
    statement = question.split("[STATEMENT]")[-1].split("[PLAN]")[0]
    match = re.search(
        r"As initial conditions I have that,\s*(.*?)\.\s*"
        r"My goal is to have that\s*(.*?)\.",
        statement,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    initial, goal_text = match.groups()
    seen: list[str] = []
    for name in re.findall(r"the ([a-z]+) block", statement, re.IGNORECASE):
        name = name.lower()
        if name not in seen:
            seen.append(name)

    on: dict[str, str] = {b: "table" for b in seen}
    for top, bottom in re.findall(
        r"the ([a-z]+) block is on top of the ([a-z]+) block",
        initial,
        re.IGNORECASE,
    ):
        on[top.lower()] = bottom.lower()
    for block in re.findall(
        r"the ([a-z]+) block is on the table",
        initial,
        re.IGNORECASE,
    ):
        on[block.lower()] = "table"

    goals = {
        top.lower(): bottom.lower()
        for top, bottom in re.findall(
            r"the ([a-z]+) block is on top of the ([a-z]+) block",
            goal_text,
            re.IGNORECASE,
        )
    }
    if not seen or not goals:
        return None

    ordered = list(goals.keys()) + [b for b in seen if b not in goals]
    return ordered, {b: on.get(b, "table") for b in ordered}, goals


def _blocks_bfs(blocks: list[str], goals: dict[str, str], start) -> list[str]:
    queue = deque([(start, [])])
    seen = {start}
    max_depth = 14

    while queue:
        state, plan = queue.popleft()
        if _blocks_goal_met(blocks, state, goals):
            return plan
        if len(plan) >= max_depth:
            continue

        for next_state, action in _blocks_next_states(blocks, goals, state):
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, plan + [action]))

    return []


def _blocks_goal_met(blocks: list[str], state, goals: dict[str, str]) -> bool:
    on_values, holding = state
    if holding is not None:
        return False
    on = dict(zip(blocks, on_values))
    return all(on.get(top) == bottom for top, bottom in goals.items())


def _blocks_next_states(blocks: list[str], goals: dict[str, str], state):
    on_values, holding = state
    on = dict(zip(blocks, on_values))
    clear = _clear_blocks(blocks, on)

    def pack(new_on: dict[str, str], new_holding):
        return (tuple(new_on[b] for b in blocks), new_holding)

    moves = []
    if holding:
        wanted = goals.get(holding)
        if wanted in clear and _support_ready(wanted, on, goals):
            new_on = dict(on)
            new_on[holding] = wanted
            moves.append((pack(new_on, None), f"(stack {holding} {wanted})"))

        new_on = dict(on)
        new_on[holding] = "table"
        moves.append((pack(new_on, None), f"(put-down {holding})"))
        return moves

    # Move clear blocks that are in the way before picking up unrelated table blocks.
    for block in _move_order(blocks, goals):
        if block in clear and on[block] not in {"table", "held"}:
            new_on = dict(on)
            bottom = new_on[block]
            new_on[block] = "held"
            moves.append((pack(new_on, block), f"(unstack {block} {bottom})"))

    for block in _move_order(blocks, goals):
        wanted = goals.get(block)
        if block in clear and on[block] == "table" and wanted in clear:
            new_on = dict(on)
            new_on[block] = "held"
            moves.append((pack(new_on, block), f"(pick-up {block})"))

    return moves


def _clear_blocks(blocks: list[str], on: dict[str, str]) -> set[str]:
    covered = {bottom for bottom in on.values() if bottom not in {"table", "held"}}
    return {b for b in blocks if b not in covered and on[b] != "held"}


def _move_order(blocks: list[str], goals: dict[str, str]) -> list[str]:
    return list(goals.keys()) + [b for b in blocks if b not in goals]


def _support_ready(block: str, on: dict[str, str], goals: dict[str, str]) -> bool:
    wanted = goals.get(block)
    return wanted is None or on.get(block) == wanted


def _solve_attack_feast_problem(question: str) -> str:
    if "Attack object" not in question or "Feast object" not in question:
        return ""

    statement = question.split("[STATEMENT]")[-1].split("[PLAN]")[0]
    match = re.search(
        r"As initial conditions I have that,\s*(.*?)\.\s*"
        r"My goal is to have that\s*(.*?)\.",
        statement,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""

    initial, goal_text = match.groups()
    objects = sorted(set(re.findall(r"object ([a-z])", statement, re.IGNORECASE)))
    goals = frozenset(
        re.findall(r"object ([a-z]) craves object ([a-z])", goal_text, re.IGNORECASE)
    )
    start = (
        frozenset(re.findall(r"province object ([a-z])", initial, re.IGNORECASE)),
        frozenset(re.findall(r"planet object ([a-z])", initial, re.IGNORECASE)),
        frozenset(re.findall(r"pain object ([a-z])", initial, re.IGNORECASE)),
        frozenset(re.findall(r"object ([a-z]) craves object ([a-z])", initial, re.IGNORECASE)),
        "harmony" in initial.lower(),
    )

    queue = deque([(start, [])])
    seen = {start}
    while queue:
        state, plan = queue.popleft()
        if goals <= state[3]:
            return "\n".join(plan) + "\n"
        if len(plan) >= 14:
            continue
        for next_state, action in _attack_feast_moves(objects, state):
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, plan + [action]))

    return ""


def _attack_feast_moves(objects: list[str], state):
    province, planet, pain, craves, harmony = state
    moves = []
    for obj, other in sorted(craves):
        if harmony and obj in province:
            new_province = set(province) - {obj}
            new_province.add(other)
            moves.append((
                (frozenset(new_province), planet, frozenset(set(pain) | {obj}), frozenset(set(craves) - {(obj, other)}), False),
                f"(feast {obj} {other})",
            ))
    for obj in objects:
        if obj in pain:
            for other in objects:
                if other in province:
                    new_province = set(province) | {obj}
                    new_province.discard(other)
                    moves.append((
                        (frozenset(new_province), planet, frozenset(set(pain) - {obj}), frozenset(set(craves) | {(obj, other)}), True),
                        f"(overcome {obj} {other})",
                    ))
    for obj in objects:
        if obj in pain:
            moves.append((
                (frozenset(set(province) | {obj}), frozenset(set(planet) | {obj}), frozenset(set(pain) - {obj}), craves, True),
                f"(succumb {obj})",
            ))
    for obj in objects:
        if harmony and obj in province and obj in planet:
            moves.append((
                (frozenset(set(province) - {obj}), frozenset(set(planet) - {obj}), frozenset(set(pain) | {obj}), craves, False),
                f"(attack {obj})",
            ))
    return moves


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
        (r"^drive ([\w\-]+) from ([\w\-]+) to ([\w\-]+)$", "(drive {0} {1} {2})"),
        (r"^use ([\w\-]+) to lift ([\w\-]+) from ([\w\-]+) at ([\w\-]+)$", "(lift {0} {1} {2} {3})"),
        (r"^use ([\w\-]+) to load ([\w\-]+) into ([\w\-]+) at ([\w\-]+)$", "(load {0} {1} {2} {3})"),
        (r"^use ([\w\-]+) to unload ([\w\-]+) from ([\w\-]+) at ([\w\-]+)$", "(unload {0} {1} {2} {3})"),
        (r"^use ([\w\-]+) to drop ([\w\-]+) to ([\w\-]+) at ([\w\-]+)$", "(drop {0} {1} {2} {3})"),
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
    fence = re.search(r"```(?:plan)?\s*\n(.*?)\n```", s, re.DOTALL | re.IGNORECASE)
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
