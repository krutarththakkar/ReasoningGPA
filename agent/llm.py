"""
LLM API wrapper.
Single responsibility ie. make HTTP calls to the BaseURl.
No core logic here, just the call, error handling, and rate limiting.
"""

from __future__ import annotations

import os
import time
import requests

API_KEY  = "sk-3X-ZR0l-AOofvpds5fhsaA"
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# delay betw calls
_RATE_SLEEP = 0.3

# Per question call counter (reset by strategies)
_call_count = 0
MAX_CALLS_PER_QUESTION = 18  # leave buffer below 20


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

    _call_count += 1
    time.sleep(_RATE_SLEEP)

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

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"] or ""
            return text.strip()
        return ""
    except Exception:
        return ""
