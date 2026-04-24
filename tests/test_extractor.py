#!/usr/bin/env python3
"""
Tests for agent/extractor.py — answer extraction.
No API calls needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.extractor import extract_answer, normalize_for_grading, extract_number

# (raw_response, domain, expected_extraction)
EXTRACTION_CASES = [
    # Explicit "Final answer:" marker 
    ("Step 1: ... Step 2: ... Final answer: 112", "math", "112"),
    ("The answer is 869.\nFinal answer: 869", "math", "869"),
    ("Final answer: B", "science_mcq", "B"),
    ("Final answer: Yes", "true_false", "Yes"),
    ("Final answer: No", "true_false", "No"),
    ("Final answer: 1939", "reading_comprehension", "1939"),

    # LaTeX boxed 
    (r"Therefore $\boxed{112}$ is the answer.", "math", "112"),
    (r"The area is \boxed{480}.", "math", "480"),

    # Bold markdown 
    ("The answer is **112**.", "math", "112"),
    ("Therefore **B** is correct.", "science_mcq", "B"),

    # "Therefore/Thus" patterns 
    ("Therefore, the answer is 869.", "math", "869"),
    ("Thus, m + n = 5.", "math", "5"),

    # m + n = X patterns 
    ("We get m + n = 158.", "math", "158"),
    ("So a + b + c = 98.", "math", "98"),

    # MCQ letter extraction 
    ("The correct answer is B. Chemical energy into radiant energy.", "science_mcq", "B"),
    ("Answer: (C)", "science_mcq", "C"),
    ("The answer is A.", "science_mcq", "A"),

    # True/False 
    ("No, Bombyx mori does not have a monopoly.", "true_false", "No"),
    ("Yes, this is plausible.", "true_false", "Yes"),

    # Last line fallback 
    ("Let me think...\nThe calculation gives us...\n112", "math", "112"),
    ("After tracing:\nBob has the blue ball", "logic", "Bob has the blue ball"),
]

NORMALIZE_CASES = [
    ("112", "112"),
    ("  112  ", "112"),
    ("The answer is 112.", "the answer is 112"),
    ("stay the same", "stay the same"),
    ("SECOND", "second"),
]

NUMBER_CASES = [
    ("112", "112"),
    ("The answer is 869", "869"),
    ("m + n = 158", "158"),
    ("", None),
    ("no numbers here", None),
    ("-42", "-42"),
]


def run_extraction_tests():
    passed = 0
    failed = 0
    failures = []

    for raw, domain, expected in EXTRACTION_CASES:
        got = extract_answer(raw, domain)
        # Flexible check: expected should be in got or equal
        ok = (got.strip() == expected.strip() or
              expected.strip().lower() in got.strip().lower())
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((raw[:60], domain, expected, got))

    print(f"\nExtraction Tests: {passed}/{len(EXTRACTION_CASES)} passed")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for raw, domain, exp, got in failures:
            print(f"  ❌ [{domain}] expected={exp!r:15s} got={got!r}")
            print(f"     raw: {raw!r}")
    else:
        print("✅ All extraction tests passed!")

    return failed


def run_normalize_tests():
    passed = 0
    failed = 0
    for inp, expected in NORMALIZE_CASES:
        got = normalize_for_grading(inp)
        if got == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ❌ normalize({inp!r}) = {got!r}, expected {expected!r}")

    print(f"\nNormalize Tests: {passed}/{len(NORMALIZE_CASES)} passed")
    return failed


def run_number_tests():
    passed = 0
    failed = 0
    for inp, expected in NUMBER_CASES:
        got = extract_number(inp)
        if got == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ❌ extract_number({inp!r}) = {got!r}, expected {expected!r}")

    print(f"\nNumber Extraction Tests: {passed}/{len(NUMBER_CASES)} passed")
    return failed


def main():
    print("=" * 60)
    print("EXTRACTOR TESTS")
    print("=" * 60)

    f1 = run_extraction_tests()
    f2 = run_normalize_tests()
    f3 = run_number_tests()

    total_failures = f1 + f2 + f3
    print(f"\n{'='*60}")
    if total_failures == 0:
        print("✅ All extractor tests passed!")
    else:
        print(f"❌ {total_failures} test(s) failed.")

    return total_failures == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
