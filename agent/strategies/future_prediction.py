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
    if re.search(
        r"(多少|price|index|rate|number|数字|价格|指数|市值|收盘价|开盘价|占有率|%|points?)",
        question,
        re.IGNORECASE,
    ):
        return "numeric"
    return "list"


def _base_rate_prediction(question: str) -> str:
    q = question.lower()
    if _question_type(question) == "options" and "number of" in q and "finish" in q:
        letters = _option_letters(question)
        if letters:
            return _as_answer([letters[-1]], question)
    return ""


def _asks_for_list(question: str) -> bool:
    return bool(
        re.search(
            r"(哪几个|哪些|是谁|名称|名字|歌曲|电影|车型|视频号|快手号|项目名|"
            r"names? only|give the names|who will be|ranked from|top\s*\d+|前\s*\d+\s*名)",
            question,
            re.IGNORECASE,
        )
    )


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

    parts = [p.strip().strip("'\"") for p in re.split(r"[,，、;；\n]+", content)]
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
    # Option prompts include examples like \boxed{A}; do not treat those as choices.
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
