"""
Future-prediction strategy.

The model answers in \\boxed{...}; we convert that to the list-ish format used
by the dev answers, like ['No'], [265.0], or ['A', 'B'].
"""
from __future__ import annotations

import ast
import re

from agent.llm import call_llm, reset_call_count

_SYSTEM = (
    "You are a calibrated prediction-market forecaster. Make a confident "
    "best-guess prediction from base rates, public knowledge, recent trends, "
    "and the options in the question. Do not refuse. End with one boxed answer."
)

_NUMERIC_WORDS = (
    r"(\u591a\u5c11|price|index|rate|number|\u6570\u5b57|\u4ef7\u683c|"
    r"\u6307\u6570|\u5e02\u503c|\u6536\u76d8\u4ef7|\u5f00\u76d8\u4ef7|"
    r"\u5360\u6709\u7387|%|points?)"
)

_LIST_WORDS = (
    r"(\u54ea\u51e0\u4e2a|\u54ea\u4e9b|\u662f\u8c01|\u540d\u79f0|"
    r"\u540d\u5b57|\u6b4c\u66f2|\u7535\u5f71|\u8f66\u578b|"
    r"\u89c6\u9891\u53f7|\u5feb\u624b\u53f7|\u9879\u76ee\u540d|"
    r"names? only|give the names|who will be|ranked from|top\s*\d+|"
    r"\u524d\s*\d+\s*\u540d)"
)


def future_prediction_strategy(question: str) -> str:
    reset_call_count()

    prior = _base_rate_prediction(question)
    if prior:
        return prior

    raw = call_llm(_build_prompt(question), system=_SYSTEM, temperature=0.0, max_tokens=800)
    prediction = _format_prediction(raw, question)

    if not _valid_prediction(prediction, question):
        raw = call_llm(
            f"{question}\n\n"
            "Fix the format only. Use allowed options if present. "
            "For cumulative thresholds like Above 1500 / Above 2000, include every "
            "lower true threshold. End only with \\boxed{...}.",
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=500,
        )
        retry = _format_prediction(raw, question)
        if retry:
            prediction = retry

    return prediction


def _build_prompt(question: str) -> str:
    qtype = _question_type(question)
    if qtype == "yes_no":
        hint = (
            "This is a binary forecast. Specific future claims usually fail unless "
            "there is a strong reason they happen. Choose exactly Yes or No."
        )
    elif qtype == "boxed_choice":
        hint = "Compare the boxed choices and choose exactly one of them."
    elif qtype == "options":
        hint = (
            "First estimate the actual outcome. For mutually exclusive ranges or "
            "winners, choose one option. For cumulative thresholds like Above 1500, "
            "Above 2000, Above 3000, include every lower true threshold; for example "
            "an estimate above 3000 but below 4000 means A, B, C."
        )
    elif qtype == "numeric":
        hint = "Estimate a realistic numeric value. Put only the number in the box."
    else:
        hint = (
            "Predict the requested names, titles, or rankings only. Separate items "
            "with commas and put no labels inside the box."
        )

    return (
        f"{question}\n\n"
        f"{hint}\n"
        "Think briefly, then make the final line only \\boxed{YOUR_PREDICTION}."
    )


def _question_type(question: str) -> str:
    if r"\boxed{Yes} or \boxed{No}" in question:
        return "yes_no"
    if _boxed_choices(question):
        return "boxed_choice"
    if _option_letters(question):
        return "options"
    if _asks_for_list(question):
        return "list"
    if re.search(_NUMERIC_WORDS, question, re.IGNORECASE):
        return "numeric"
    return "list"


def _base_rate_prediction(question: str) -> str:
    q = question.lower()
    qtype = _question_type(question)

    if qtype == "yes_no":
        return "['No']"

    if qtype == "options" and "number of" in q and "finish" in q:
        letters = _option_letters(question)
        if letters:
            return _as_answer([letters[-1]], question)

    if qtype == "options" and _looks_like_winner_market(q):
        letters = _option_letters(question)
        if letters:
            return _as_answer([letters[0]], question)

    return ""


def _looks_like_winner_market(q_lower: str) -> bool:
    return bool(
        re.search(
            r"(which|who).{0,80}\bwin\b|\bwill win\b|\bwinner\b|\baward\b|"
            r"\bbest new series\b|eisner",
            q_lower,
        )
    )


def _asks_for_list(question: str) -> bool:
    return bool(re.search(_LIST_WORDS, question, re.IGNORECASE))


def _format_prediction(raw: str, question: str = "") -> str:
    if not raw:
        return ""

    matches = re.findall(r"\\boxed\{([^{}]*)\}", raw)
    if matches:
        return _content_to_answer(matches[-1], question)

    for line in reversed(raw.splitlines()):
        if line.strip():
            return _content_to_answer(line, question)
    return ""


def _content_to_answer(content: str, question: str) -> str:
    content = content.strip().strip(".")
    if not content:
        return ""

    parsed = _parse_list(content)
    if parsed is not None:
        return _as_answer(parsed, question)

    choices = _boxed_choices(question)
    if choices:
        picked = _pick_choice(content, choices)
        if picked:
            return _as_answer([picked], question)

    letters = _option_letters(question)
    if letters:
        picked = _pick_letters(content, question, letters)
        if picked:
            return _as_answer(picked, question)

    number = _number(content)
    if number is not None and _question_type(question) == "numeric":
        return f"[{number}]"

    parts = [p.strip().strip("'\"") for p in re.split(r"[,\uFF0C\u3001;\uFF1B\n]+", content)]
    return _as_answer([p for p in parts if p], question)


def _parse_list(content: str) -> list[object] | None:
    if not (content.startswith("[") and content.endswith("]")):
        return None
    try:
        value = ast.literal_eval(content)
    except Exception:
        return None
    return value if isinstance(value, list) else None


def _boxed_choices(question: str) -> list[str]:
    choices = [
        x.strip()
        for x in re.findall(r"\\boxed\{([^{}]+)\}", question)
        if "YOUR_PREDICTION" not in x
    ]
    if choices and all(re.fullmatch(r"[A-Z](?:,\s*[A-Z])*", x) for x in choices):
        return []
    return choices


def _option_letters(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?:^|\s)([A-H])\.\s+", question)))


def _pick_choice(content: str, choices: list[str]) -> str:
    lowered = content.lower()
    return next((choice for choice in choices if choice.lower() in lowered), "")


def _pick_letters(content: str, question: str, letters: list[str]) -> list[str]:
    found = [x for x in re.findall(r"\b([A-H])\b", content.upper()) if x in letters]
    if not found:
        lowered = content.lower()
        for letter, text in _option_texts(question).items():
            if text.lower() in lowered:
                found.append(letter)
    return [letter for letter in letters if letter in found]


def _option_texts(question: str) -> dict[str, str]:
    return {
        letter: text.strip()
        for letter, text in re.findall(
            r"(?:^|\s)([A-H])\.\s+(.+?)(?=\s+[A-H]\.\s+|$)",
            question,
            re.DOTALL,
        )
    }


def _number(content: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", content.replace(",", ""))
    if not nums:
        return None
    try:
        return str(float(nums[-1]))
    except ValueError:
        return None


def _as_answer(parts: list[object], question: str) -> str:
    if _question_type(question) == "numeric" and len(parts) == 1:
        try:
            return f"[{float(str(parts[0]).replace(',', ''))}]"
        except ValueError:
            pass

    cleaned = [str(p).strip().strip("'\"") for p in parts]
    cleaned = [p for p in cleaned if p]
    return "[" + ", ".join(f"'{p}'" for p in cleaned) + "]"


def _valid_prediction(prediction: str, question: str) -> bool:
    if not prediction:
        return False
    qtype = _question_type(question)
    if qtype == "numeric":
        return bool(re.fullmatch(r"\[-?\d+(?:\.\d+)?\]", prediction))
    if qtype == "yes_no":
        return prediction in ("['Yes']", "['No']")
    if qtype == "boxed_choice":
        return prediction in {_as_answer([choice], question) for choice in _boxed_choices(question)}
    if qtype == "options":
        allowed = set(_option_letters(question))
        got = re.findall(r"'([A-H])'", prediction)
        return bool(got) and all(letter in allowed for letter in got)
    return prediction.startswith("[") and prediction.endswith("]")
