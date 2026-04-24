# Evaluation Strategy

## How the Grader Works (Our Best Guess)

The tutorial shows two grading approaches:

### 1. Exact Match After Normalization
```python
def normalize_text(s):
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s
```
If `normalize(prediction) == normalize(expected)` → correct.

### 2. LLM-as-Judge
The model is asked: "Is PREDICTION correct for EXPECTED_ANSWER? Reply True or False."
This is flexible — "112" and "the answer is 112" both pass.

### What This Means for Us
- We don't need to be perfectly terse
- But we should avoid long explanations in the output — they can confuse the judge
- Numbers should be clean: "112" not "112.0" or "one hundred twelve"
- For MCQ: just the letter "B" or "B. chemical energy" — both should pass
- For true/false: "Yes" or "No" — not "Yes, because..."

---

## Our Eval Script Design

### `eval/run_eval.py`

```
Usage:
  python eval/run_eval.py                    # all 1000 dev questions
  python eval/run_eval.py --n 50             # first 50
  python eval/run_eval.py --n 50 --start 100 # questions 100-149
  python eval/run_eval.py --domain math      # only math questions
  python eval/run_eval.py --sample fixed     # always same 50 questions
```

Output:
- Prints per-domain accuracy table
- Saves full results to `eval/results/TIMESTAMP.json`
- Saves summary to `eval/results/latest_summary.json`

### `eval/grader.py`

Grading priority:
1. Exact match after normalization → True (0 extra LLM calls)
2. Numeric extraction match → True (0 extra LLM calls)
3. LLM-as-judge → True/False (1 extra LLM call)

We use LLM-as-judge sparingly — only when exact match fails.

### `eval/analyze_failures.py`

```
Usage:
  python eval/analyze_failures.py eval/results/latest.json
  python eval/analyze_failures.py eval/results/latest.json --domain math --top 10
```

Output:
- Groups failures by domain
- Shows question, expected, got for each failure
- Categorizes failure type (wrong answer / extraction failure / empty)

---

## Fixed 50-Question Sample

For consistent comparison across runs, we always use the same 50 questions.
These are selected to cover all domains proportionally:
- 15 math
- 10 word_problem
- 8 reading_comprehension
- 7 science_mcq
- 5 logic
- 3 true_false
- 2 commonsense

Stored in `eval/fixed_sample_indices.json`.

---

## Accuracy Targets

| Domain | Baseline (1 call, no technique) | Target (with techniques) |
|--------|--------------------------------|--------------------------|
| math | ~20% | ≥ 45% |
| word_problem | ~50% | ≥ 75% |
| reading_comprehension | ~70% | ≥ 85% |
| science_mcq | ~75% | ≥ 88% |
| logic | ~60% | ≥ 80% |
| true_false | ~80% | ≥ 90% |
| commonsense | ~70% | ≥ 80% |
| **overall** | **~50%** | **≥ 70%** |

---

## Tracking Progress

After each milestone, record results in this table:

| Milestone | Date | Overall | Math | Word | RC | MCQ | Logic | T/F | CS |
|-----------|------|---------|------|------|----|-----|-------|-----|----|
| M2 baseline | - | - | - | - | - | - | - | - | - |
| M3 math fix | - | - | - | - | - | - | - | - | - |
| M4 word fix | - | - | - | - | - | - | - | - | - |
| M5 easy fix | - | - | - | - | - | - | - | - | - |
| M7 full eval | - | - | - | - | - | - | - | - | - |

---

## Evaluation Cost JustForFun

Each eval run costs LLM calls:
- 50 questions × ~2 calls avg = ~100 calls for agent
- 50 questions × 1 call for LLM judge (only on failures) = ~25 calls
- Total: ~125 calls per 50-question eval run
