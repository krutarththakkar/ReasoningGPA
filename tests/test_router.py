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
    (
        "Let $ABCD$ be a convex quadrilateral with $AB = CD = 10$. Find the area.",
        "math"
    ),
    (
        r"What is the product of the real roots of $x^2 + 18x + 30 = 2\sqrt{x^2+18x+45}$?",
        "math"
    ),
    (
        "Compute $\\sqrt{(31)(30)(29)(28)+1}$.",
        "math"
    ),
    (
        "How many even integers between 4000 and 7000 have four different digits?",
        "math"
    ),
    (
        "Find the remainder when $\\binom{\\binom{3}{2}}{2}$ is divided by 1000.",
        "math"
    ),

    # --- WORD PROBLEM ---
    (
        "Jane has saved $4 of her allowance every week for the past 8 weeks. "
        "How many more weeks will it take to save $60?",
        "word_problem"
    ),
    (
        "A marketing company pays 30% commission on sales up to $1000. "
        "Calculate how much Antonella earned selling $2500 worth of goods.",
        "word_problem"
    ),
    (
        "Daisy bought potatoes weighing 5 pounds and sweet potatoes weighing 2 times as much. "
        "How many pounds of carrots did she buy if carrots weigh 3 pounds fewer than sweet potatoes?",
        "word_problem"
    ),
    (
        "Rory makes a cake that weighs 20 ounces. She cuts it into 8 pieces. "
        "Rory and her mom each have a piece. How much does the remaining cake weigh?",
        "word_problem"
    ),

    # --- READING COMPREHENSION ---
    (
        "The Tower Theatre was built in 1939. Context: The popular neighborhood known as "
        "the Tower District is centered around the historic Tower Theatre. "
        "When was the Tower Theatre built?",
        "reading_comprehension"
    ),
    (
        "Which record label released Van Morrison's song? Context: His Band and the Street Choir "
        "was released on 15 November 1970 by Warner Bros. Records. [PAR] The album was produced...",
        "reading_comprehension"
    ),

    # --- SCIENCE MCQ ---
    (
        "Which best describes the transformation of energy in a flashlight? "
        "A. chemical energy into sound energy "
        "B. chemical energy into radiant energy "
        "C. electrical energy into nuclear energy "
        "D. electrical energy into mechanical energy",
        "science_mcq"
    ),
    (
        "A student walks to school and notices the grass is wet but streets are dry. "
        "Which process caused this? A. condensation B. erosion C. evaporation D. precipitation",
        "science_mcq"
    ),
    (
        "Which property makes the water cycle on Earth possible? "
        "A. Water can change phase. B. Water can be absorbed by plants. "
        "C. Water is important in chemical reactions. D. Water is the universal solvent.",
        "science_mcq"
    ),

    # --- LOGIC ---
    (
        "Alice, Bob, and Claire are playing a game. Alice has a green ball, Bob has a black ball. "
        "First, Bob and Alice swap balls. Then, Claire and Alice swap balls. "
        "Finally, Alice and Bob swap balls. At the end of the game, Bob has the\n"
        "Options:\n(A) green ball\n(B) black ball\n(C) blue ball",
        "logic"
    ),
    (
        "In a race, you pass the person in second place. What position are you now in? "
        "Options:\n(A) first\n(B) second\n(C) third",
        "logic"
    ),

    # --- TRUE/FALSE ---
    (
        "Does Bombyx mori have a monopoly over silk production? "
        "Facts: A monopoly refers to the exclusive supply of a good. "
        "Spiders, beetles, caterpillars, and fleas produce silk.",
        "true_false"
    ),
    (
        "Is the Louvre's pyramid known for being unbreakable? "
        "Facts: The Pyramid at the Louvre is made of glass and metal. "
        "10mm thick glass is not unbreakable.",
        "true_false"
    ),
    (
        "Is the following sentence plausible? "
        "\"Miro Heiskanen earned a trip to the penalty box.\"",
        "true_false"
    ),

    # --- COMMONSENSE ---
    (
        "Find a movie similar to The Fugitive, Terminator 2, Aladdin, Toy Story:\n"
        "Options:\n(A) The Edge of Love\n(B) Untitled Spider-Man Reboot\n"
        "(C) The Lion King\n(D) Daddy Day Camp",
        "commonsense"
    ),

    # --- Dev-data domains (coding / planning / future_prediction) ---
    ("Retrieves names. The function should output list. Write self-contained code starting with def task_func(user):", "coding"),
    ("I am playing with a set of objects. Here are the actions I can do", "planning"),
    ("You are an agent that can predict future events. \\boxed{YOUR_PREDICTION}", "future_prediction"),
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
