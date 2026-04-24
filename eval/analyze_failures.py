#!/usr/bin/env python3
"""
Failure analysis tool — helps identify what to fix next.
"""

import argparse
import json
from pathlib import Path

def analyze(results_path: Path, domain_filter: str = None, top_n: int = 10) -> None:
    with results_path.open() as f:
        data = json.load(f)

    results = data.get("results", [])
    failures = [r for r in results if not r["correct"]]

    if domain_filter:
        failures = [r for r in failures if r["domain"] == domain_filter]

    print(f"\n{'='*65}")
    print(f"FAILURE ANALYSIS — {results_path.name}")
    print(f"Total failures: {len(failures)} / {len(results)}")
    print(f"{'='*65}")

    # output: list failures
    for i, f in enumerate(failures[:top_n], 1):
        print(f"\n[{i}] Domain: {f['domain']}")
        print(f"  Q: {f['question'][:100]}...")
        print(f"  Expected: {f['expected']!r}")
        print(f"  Got:      {f['prediction']!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--top",    type=int, default=10)
    args = parser.parse_args()

    analyze(Path(args.results_file), args.domain, args.top)
