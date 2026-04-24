# CSE 476 Final Project — Master Plan

## Project Goal

Build a general-purpose reasoning agent that:
- Solves diverse question types (math, reading comprehension, logic, science, word problems, etc.)
- Uses ≥ 8 distinct inference-time techniques
- Stays under 20 LLM calls per question
- Produces clean string answers that match the grader's expectations

---

## What We Know About the Grader

From the tutorial notebook:
- Grader uses **exact string match after normalization** (lowercase, strip punctuation)
- Grader also uses **LLM-as-judge** for flexible matching (so "112" and "the answer is 112" both pass)
- Answers must be strings, under 5000 characters
- The model itself (`qwen3-30b-a3b-instruct-2507`) is used as the judge

Key implication: **we don't need to be perfectly terse — we need to be correct**. The judge is flexible. But shorter, cleaner answers are safer.

---

## What We Know About the Data

### Dev data (1000 questions, labeled by domain):
- **math** — AIME-style competition problems. Hard. Require multi-step reasoning. Answers are integers or simple expressions.
- **word_problem** — Arithmetic in natural language. Sometimes multilingual (Spanish, Chinese). Answers are numbers.
- **reading_comprehension** — Long context passage + question. Answer is a span from the passage.
- **science_mcq** — Multiple choice (A/B/C/D). Science concepts.
- **logic** — State tracking, ball swapping, ordering puzzles. Options given.
- **true_false** — Facts provided, evaluate a claim. Answer is Yes/No.
- **commonsense** — General knowledge, plausibility, similarity. Short answers.

### Test data (1000+ questions, no labels):
- Same distribution as dev data
- We must predict the domain ourselves

---

## Core Insight: What Fails and Why

The tutorial notebook itself shows the baseline model fails math with `max_tokens=128`.
The model got `4` instead of `8` for `3n + 5 > 26` — it couldn't show its work.

**Root causes of failures:**
1. `max_tokens` too small — model truncates before finishing reasoning
2. No domain-specific prompting — same prompt for AIME math and reading comprehension
3. Answer extraction too naive — model says "the answer is 112" but extractor misses it
4. Self-consistency at high temperature for math — introduces noise, not signal
5. Too many LLM calls for simple questions — wastes budget on easy cases

---

## The 8 Inference-Time Techniques

| # | Technique | Description | Domain |
|---|-----------|-------------|--------|
| 1 | **Chain-of-Thought (CoT)** | Ask model to reason step by step before answering | All |
| 2 | **Domain-Aware Routing** | Classify question type, apply best strategy | Entry point |
| 3 | **Step-Back Prompting** | Ask for relevant principles/formulas first, then solve | Hard math |
| 4 | **Least-to-Most Decomposition** | Break problem into sub-problems, solve sequentially | Word problems |
| 5 | **Few-Shot Exemplars** | Include 1-2 solved examples in the prompt | MCQ, true/false |
| 6 | **Self-Refine** | After getting answer, ask model to check and correct it | Math (on failure) |
| 7 | **Answer Extraction + Normalization** | Post-process raw output to get clean final answer | All |
| 8 | **LLM-as-Judge (Self-Eval)** | Use model to verify answer correctness | Math verification |

These are all applied **within the agent loop** — no external tools :( , no paid APIs.

---

## Call Budget Per Domain

| Domain | Strategy | Calls |
|--------|----------|-------|
| math (hard) | Step-Back CoT → Extract → Self-Refine if needed | 2–3 |
| word_problem | Decomposition CoT → Extract | 1–2 |
| reading_comprehension | Direct extraction prompt | 1 |
| science_mcq | Few-shot CoT → letter extraction | 1 |
| logic | Step-by-step trace CoT | 1 |
| true_false | Few-shot + fact evaluation | 1 |
| commonsense | CoT | 1 |

**Maximum: 3 calls per question. Well under the 20-call limit.**

---

## Iterative Development Strategy

### The Loop

```
Write code → Run eval on 50 dev questions → Analyze failures → Fix → Repeat
```

Each iteration should:
1. Run eval on a fixed 50-question sample (same questions each time for comparability)
2. Record per-domain accuracy
3. Identify the top failure pattern
4. Fix exactly that one thing
5. Re-run and confirm improvement

### Stopping Criteria
- Dev accuracy ≥ 70% overall
- No domain below 50%
- All techniques implemented and documented

---

## File Structure

See `docs/STRUCTURE.md` for the full project layout.

---

## Milestones

See `docs/MILESTONES.md` for the step-by-step build plan.

---

## Evaluation Strategy

See `docs/EVALUATION.md` for how we measure progress.
