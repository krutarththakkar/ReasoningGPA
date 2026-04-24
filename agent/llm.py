"""
LLM API wrapper.
Single responsibility ie. make HTTP calls to the BaseURl.
No core logic here, just the call, error handling, and rate limiting.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from repo-root .env without a dependency."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

API_KEY  = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE = (
    os.getenv("API_BASE")
    or os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or "https://openai.rc.asu.edu/v1"
)
MODEL    = (
    os.getenv("MODEL")
    or os.getenv("MODEL_NAME")
    or "qwen3-30b-a3b-instruct-2507"
)

# delay betw calls
_RATE_SLEEP = 0.3

# Per question call counter (reset by strategies)
_call_count = 0
MAX_CALLS_PER_QUESTION = 18  # leave buffer below 20
_warned: set[str] = set()


def _debug(message: str) -> None:
    if os.getenv("LLM_DEBUG") == "1":
        print(f"[llm] {message}", file=sys.stderr)


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"[llm] WARNING: {message}", file=sys.stderr)


def is_configured() -> bool:
    return bool(API_KEY)


def reset_call_count() -> None:
    global _call_count
    _call_count = 0


def get_call_count() -> int:
    return _call_count


def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 90,
) -> str:
    """
    Call the LLM and return the response text.
    Returns empty string on any failure — never raises.
    """
    global _call_count

    if _call_count >= MAX_CALLS_PER_QUESTION:
        return ""

    if not API_KEY:
        _warn_once(
            "missing-api-key",
            "API_KEY/OPENAI_API_KEY is not set. LLM-backed strategies will return empty answers.",
        )
        return ""

    _call_count += 1
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(2):
        try:
            time.sleep(_RATE_SLEEP)
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                _debug(f"HTTP {resp.status_code} on attempt {attempt + 1}")
                _warn_once(
                    "http-error",
                    f"LLM API returned HTTP {resp.status_code}. Check API_BASE, MODEL, and API key.",
                )
                continue

            text = resp.json()["choices"][0]["message"]["content"] or ""
            text = text.strip()
            if text:
                return text
            _debug(f"empty response on attempt {attempt + 1}")
        except Exception as exc:
            _debug(f"request failed on attempt {attempt + 1}: {type(exc).__name__}")
            _warn_once(
                "request-failed",
                f"LLM request failed with {type(exc).__name__}. Check API_BASE/network settings.",
            )

    return ""
