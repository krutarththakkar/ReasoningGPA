# Build Milestones

## Philosophy: Build → Measure → Fix → Repeat

Each milestone produces something runnable and measurable.
Never move to the next milestone until the current one passes its acceptance test.

---

## Milestone 0 — Baseline (already done, broken)

**What exists**: `agent.py` monolith with all techniques mixed together.

**Problems identified**:
- `max_tokens=512` too small for math
- Answer extractor misses many patterns
- Self-consistency at temperature=0.7 adds noise to math
- Too many calls for simple questions
- No modular structure

**Decision**: Rewrote from scratch following the modular structure in `STRUCTURE.md`.

---

## Milestone 1 — Foundation (LLM wrapper + router + extractor)

**Goal**: The three zero-logic-but-critical modules work correctly.

### Tasks
- [x] Create `agent/` package structure
- [x] Write `agent/llm.py` — API wrapper with call counting
- [x] Write `agent/router.py` — domain detection (heuristics only, 0 LLM calls)
- [x] Write `agent/extractor.py` — answer extraction for all formats

### Acceptance Test
Run `python -m pytest agent/tests/test_router.py` — all domain classifications correct on 20 hand-picked examples.
Run `python -m pytest agent/tests/test_extractor.py` — all extraction patterns work.

### No LLM calls needed for this milestone.

---

## Milestone 2 — Simplest Possible Agent (1 call per question)

**Goal**: A working agent that uses 1 LLM call per question with a good CoT prompt.
This is the baseline we measure everything against.

### Tasks
- [ ] Write `agent/techniques/cot.py`
- [ ] Write one strategy per domain (all using CoT, no fancy techniques yet)
- [ ] Wire up `agent/__init__.py` with `agent_loop()`
- [ ] Update `generate_answer_template.py` to import from `agent/`

### Acceptance Test
Run `python eval/run_eval.py --n 50 --sample fixed_50` and record baseline accuracy.
Expected: ~40-50% overall (math will be low, easy domains will be high).

### Call budget: exactly 1 per question.

---

## Milestone 3 — Fix Math (the hardest domain)

**Goal**: Improve math accuracy from ~20% to ~50%+.

### Why math fails
- Needs more tokens to show work
- Needs structured prompting (step-back: principles first, then solve)
- Needs self-refine when answer looks wrong

### Tasks
- [ ] Write `agent/techniques/step_back.py`
- [ ] Write `agent/techniques/self_refine.py`
- [ ] Write `agent/strategies/math_strategy.py` using step-back + self-refine
- [ ] Tune `max_tokens` for math (target: 1200)
- [ ] Tune answer extraction for math-specific formats (\\boxed{}, "m+n=", etc.)

### Acceptance Test
Run `python eval/run_eval.py --n 50 --domain math` — math accuracy ≥ 45%.

### Call budget: 2-3 per math question.

---

## Milestone 4 — Fix Word Problems

**Goal**: Word problem accuracy from ~50% to ~80%+.

### Why word problems fail
- Multi-step arithmetic needs explicit decomposition
- Multilingual questions (Spanish, Chinese) need language-aware prompting
- Numbers in text need careful extraction

### Tasks
- [ ] Write `agent/techniques/decompose.py`
- [ ] Write `agent/strategies/word_problem.py` using decomposition
- [ ] Add multilingual handling (detect non-English, add translation hint to prompt)
- [ ] Tune number extraction in `extractor.py`

### Acceptance Test
Run `python eval/run_eval.py --n 50 --domain word_problem` — accuracy ≥ 75%.

### Call budget: 1-2 per word problem.

---

## Milestone 5 — Fix Easy Domains (MCQ, True/False, Reading Comp)

**Goal**: These should be near-perfect with good prompting. Target ≥ 85% each.

### Tasks
- [ ] Write `agent/techniques/few_shot.py` with domain-specific examples
- [ ] Write `agent/strategies/mcq.py` — few-shot + letter extraction
- [ ] Write `agent/strategies/true_false.py` — few-shot + yes/no extraction
- [ ] Write `agent/strategies/reading_comp.py` — direct span extraction
- [ ] Write `agent/strategies/logic.py` — step-by-step state trace
- [ ] Write `agent/strategies/commonsense.py` — CoT

### Acceptance Test
Run `python eval/run_eval.py --n 50 --domain science_mcq` — ≥ 85%
Run `python eval/run_eval.py --n 50 --domain true_false` — ≥ 85%
Run `python eval/run_eval.py --n 50 --domain reading_comprehension` — ≥ 80%

### Call budget: 1 per question for all these domains.

---

## Milestone 6 — Eval Infrastructure

**Goal**: Solid eval tooling so we can measure progress reliably.

### Tasks
- [ ] Write `eval/run_eval.py` — configurable eval runner
- [ ] Write `eval/grader.py` — exact match + LLM judge
- [ ] Write `eval/analyze_failures.py` — failure pattern analysis
- [ ] Create `eval/results/` directory with gitkeep
- [ ] Define a fixed 50-question sample for consistent comparison across runs

### Acceptance Test
`python eval/run_eval.py --n 50` produces a JSON results file and prints per-domain accuracy.

---

## Milestone 7 — Full Dev Set Eval + Tuning

**Goal**: Run on all 1000 dev questions, identify remaining failure patterns, fix them.

### Tasks
- [ ] Run `python eval/run_eval.py --n 1000`
- [ ] Run `python eval/analyze_failures.py eval/results/latest.json`
- [ ] Fix top 3 failure patterns
- [ ] Re-run and confirm improvement

### Acceptance Test
Overall dev accuracy ≥ 65%.

---

## Milestone 8 — Generate Test Answers + Submit

**Goal**: Run agent on test data, validate format, submit.

### Tasks
- [ ] Run `python generate_answer_template.py`
- [ ] Validate output format (all 1000+ answers, each under 5000 chars)
- [ ] Push to GitHub with clean commit history
- [ ] Write `report.md` (one page)

### Acceptance Test
`python generate_answer_template.py` completes without errors and produces valid JSON.

---

## Failure Analysis Protocol

After each eval run, follow this process:

1. **Sort failures by domain** — which domain has the most failures?
2. **Look at the top 5 failures in that domain** — what pattern do they share?
3. **Categorize the failure**:
   - Wrong answer (model reasoned incorrectly)
   - Right answer, wrong extraction (model got it right but extractor missed it)
   - Timeout / empty response (API issue)
   - Wrong domain classification (router sent it to wrong strategy)
4. **Fix the most common category first**
5. **Re-run the same 50 questions** to confirm improvement

### Common Failure Patterns to Watch For

| Pattern | Fix |
|---------|-----|
| Model gives correct reasoning but extractor misses the number | Improve extractor regex |
| Model says "m+n = 112" but expected is just "112" | Add m+n pattern to extractor |
| Math answer is correct but has wrong sign | Check problem statement parsing |
| Reading comp answer is a long sentence instead of a span | Tighten the prompt |
| MCQ answer is "B. chemical energy" instead of "B" | Add letter-only extraction |
| Word problem in Spanish returns wrong number | Add language detection + hint |
| Logic puzzle answer is "(C) blue ball" instead of "blue ball" | Normalize option format |
