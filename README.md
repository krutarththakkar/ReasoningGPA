# Project Structure

## Directory Layout

```
NLP/
│
│
├── agent/                         # Core agent modules (one file per concern)
│   ├── __init__.py                # Exports agent_loop()
│   ├── llm.py                     # LLM API wrapper (call_llm, rate limiting)
│   ├── router.py                  # Domain detection and strategy routing
│   ├── extractor.py               # Answer extraction and normalization
│   ├── techniques/                # One file per inference-time technique
│   │   ├── __init__.py
│   │   ├── cot.py                 # Chain-of-Thought
│   │   ├── step_back.py           # Step-Back Prompting
│   │   ├── decompose.py           # Least-to-Most Decomposition
│   │   ├── few_shot.py            # Few-Shot Exemplars
│   │   ├── self_refine.py         # Self-Refine loop
│   │   └── self_eval.py           # LLM-as-Judge / Self-Evaluation
│   └── strategies/                # Per-domain strategy orchestration
│       ├── __init__.py
│       ├── math_strategy.py       # Hard math: step-back + self-refine
│       ├── word_problem.py        # Word problems: decomposition
│       ├── reading_comp.py        # Reading comprehension: direct extraction
│       ├── mcq.py                 # Multiple choice: few-shot CoT
│       ├── logic.py               # Logic/tracking: step-by-step trace
│       ├── true_false.py          # True/False: fact evaluation
│       └── commonsense.py         # Commonsense: CoT
│
├── eval/                          # Evaluation scripts
│   ├── run_eval.py                # Main eval runner (configurable)
│   ├── grader.py                  # Grading logic (exact match + LLM judge)
│   └── results/                   # Saved eval results (gitignored)
│       └── .gitkeep
│
├── data/                          # Data files (symlinks or copies)
│   ├── dev_data.json              # → cse476_final_project_dev_data.json
│   └── test_data.json             # → cse_476_final_project_test_data.json
│
├── generate_answer_template.py    # Submission runner (calls agent_loop)
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and setup instructions
└── report.md                      # One-page report (deliverable)
```

---

## Module Responsibilities

### `agent/llm.py`
- Single function: `call_llm(prompt, system, temperature, max_tokens)`
- Handles: API key, base URL, model name from env vars
- Handles: rate limiting (sleep between calls)
- Handles: error recovery (return empty string on failure, don't crash)
- Tracks: call count per question (for budget enforcement)
- **No business logic here** — just the HTTP call

### `agent/router.py`
- Single function: `detect_domain(question: str) -> str`
- Uses **pure heuristics** — no LLM call
- Returns one of: `math`, `word_problem`, `reading_comprehension`, `science_mcq`, `logic`, `true_false`, `commonsense`
- Heuristics based on: LaTeX patterns, keyword presence, question length, option format

### `agent/extractor.py`
- Single function: `extract_answer(raw: str, domain: str) -> str`
- Handles all output formats the model might produce:
  - "Final answer: 112"
  - "The answer is **112**"
  - "\\boxed{112}"
  - "Therefore, 112"
  - "(C)" or just "C"
  - "Yes" / "No"
- Falls back to last short line if no pattern matches

### `agent/techniques/`
Each file has one function that takes a question and returns a raw LLM response string.
- `cot.py` → `chain_of_thought(question, domain_hint) -> str`
- `step_back.py` → `step_back(question) -> str`
- `decompose.py` → `decompose(question) -> str`
- `few_shot.py` → `few_shot(question, domain) -> str`
- `self_refine.py` → `self_refine(question, initial_answer, domain) -> str`
- `self_eval.py` → `self_evaluate(question, answer) -> bool`

### `agent/strategies/`
Each file orchestrates techniques for one domain.
- Takes a question string
- Calls techniques in the right order
- Returns a clean answer string
- Manages call budget (stays under limit)

### `agent/__init__.py`
```python
from agent.router import detect_domain
from agent.strategies import get_strategy

def agent_loop(question: str) -> str:
    domain = detect_domain(question)
    strategy = get_strategy(domain)
    return strategy(question)
```

### `eval/run_eval.py`
- Loads dev data
- Runs agent_loop on each question
- Calls grader
- Reports per-domain accuracy
- Saves results to `eval/results/`

### `eval/grader.py`
- `grade(question, prediction, expected) -> bool`
- First tries exact match after normalization
- Then tries numeric extraction match
- Then uses LLM-as-judge (1 extra call)

---

## Design Principles

1. **One concern per file** — easy to test, easy to swap out
2. **No LLM calls in router or extractor** — these must be fast and free
3. **Strategies are the only place that orchestrate multiple techniques**
4. **Every technique is independently testable**
5. **Eval is completely separate from agent** — no coupling
6. **All config (API key, model, base URL) comes from environment variables**
