#!/usr/bin/env python3
"""
Tests for agent/router.py — domain classification.
No API calls needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.router import detect_domain

# (question_snippet, expected_domain)
TEST_CASES = [
    # --- MATH ---

    # --- WORD PROBLEM ---

    # --- READING COMPREHENSION ---

    # --- SCIENCE MCQ ---

    # --- LOGIC ---

    # --- TRUE/FALSE ---

    # --- COMMONSENSE ---
]


def run_tests():
    passed = 0
    failed = 0
    failures = []

    for question, expected in TEST_CASES:
        got = detect_domain(question)
        if got == expected:
            passed += 1
        else:
            failed += 1
            failures.append((question[:80], expected, got))

    print(f"\nRouter Tests: {passed}/{len(TEST_CASES)} passed")
    print("=" * 60)

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for q, exp, got in failures:
            print(f"  ❌ expected={exp!r:25s} got={got!r}")
            print(f"     Q: {q}...")
    else:
        print("✅ All router tests passed!")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
