# CSE 476 Final Project — Inference-Time Reasoning Agent

## Setup

```bash
pip install -r requirements.txt
```

Create .env file

```bash
# Set your API key (get from Voyager Portal)
API_KEY="your_key_here"
API_BASE="https://openai.rc.asu.edu/v1"
MODEL_NAME="qwen3-30b-a3b-instruct-2507"
LLM_DEBUG=0
```

## Agent Architecture

The agent aims to implement 9 inference-time techniques:

| #   | Technique                       | Description                                                                                                                                       |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Domain-Aware Routing**        | Classifies question type (math, word_problem, reading_comprehension, science_mcq, logic, commonsense, true_false) and routes to the best strategy |
| 2   | **Chain-of-Thought (CoT)**      | Step-by-step reasoning with explicit "Final answer:" extraction                                                                                   |
| 3   | **Self-Consistency**            | Samples 3 answers at temperature=0.7, returns majority vote (math, logic)                                                                         |
| 4   | **Step-Back Prompting**         | Derives relevant mathematical principles first, then solves (hard math)                                                                           |
| 5   | **Least-to-Most Decomposition** | Breaks multi-step word problems into sub-problems                                                                                                 |
| 6   | **Few-Shot Exemplars**          | Injects domain-specific in-context examples                                                                                                       |
| 7   | **Reflection / Self-Critique**  | Re-examines initial answer and corrects if needed                                                                                                 |
| 8   | **Verification Pass**           | Checks whether the answer satisfies question constraints                                                                                          |
| 9   | **Answer Extraction**           | Post-processes raw LLM output to return a clean final answer                                                                                      |

## Strategy per Domain

| Domain                  | Strategy                                                              |
| ----------------------- | --------------------------------------------------------------------- |
| `math`                  | Step-Back → Self-Consistency (3 samples) → Reflection if disagreement |
| `word_problem`          | Decomposition + CoT → Reflection if disagreement                      |
| `reading_comprehension` | Few-Shot + CoT                                                        |
| `science_mcq`           | Few-Shot + CoT → Reflection for letter extraction                     |
| `logic`                 | Self-Consistency (3 samples) → CoT fallback                           |
| `true_false`            | Few-Shot + CoT                                                        |
| `commonsense`           | Decomposition → Verification → Reflection if unverified               |

## LLM Call Budget

- Math: ~5–6 calls (1 classify + 1 step-back + 3 self-consistency + 1 reflect)
- Word problem: ~4 calls (1 classify + 1 decompose + 1 CoT + 1 reflect)
- Reading comprehension: ~3 calls (1 classify + 1 few-shot + 1 CoT)
- Science MCQ: ~3–4 calls
- Logic: ~4 calls
- True/False: ~3 calls
- Commonsense: ~3–4 calls

All well within the 20-call limit.

## Files

```
agent/llm.py                          # Main agent implementation
generate_answer_template.py           # Submission runner
eval/run_eval.py                      # Dev set evaluation script
requirements.txt                      # Python dependencies
cse476_final_project_dev_data.json    # Dev data (1000 questions with answers)
cse_476_final_project_test_data.json  # Test data (no answers)
cse_476_final_project_answers.json    # Output answers (generated)
```
