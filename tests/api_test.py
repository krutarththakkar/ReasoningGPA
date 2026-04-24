#!/usr/bin/env python3
"""Minimal API connectivity test."""
import requests, json, sys

API_KEY  = "yourkeyhere"
API_BASE = "https://openai.rc.asu.edu/v1"
MODEL    = "qwen3-30b-a3b-instruct-2507"

print(f"Testing API at {API_BASE} with model {MODEL}")
print(f"Key: {API_KEY[:8]}...")

try:
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=30,
    )
    print(f"HTTP Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        print(f"Answer: {answer!r}")
        print("API OK!")
    else:
        print(f"Error body: {resp.text[:500]}")
        sys.exit(1)
except Exception as e:
    print(f"Connection error: {e}")
    sys.exit(1)
