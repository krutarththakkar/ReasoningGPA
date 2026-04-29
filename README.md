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

The agent implements 10 inference-time techniques:

| #   | Technique                       | Description                                                                                                                                                                              |
| --- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Domain-Aware Routing**        | Classifies question type (math, word_problem, reading_comprehension, science_mcq, logic, commonsense, true_false, coding, planning, expression_puzzle, future_prediction) via heuristics |
| 2   | **Chain-of-Thought (CoT)**      | Step-by-step reasoning with explicit "Final answer:" extraction                                                                                                                          |
| 3   | **Self-Consistency**            | Samples 3 answers at temperature=0.7, returns majority vote (math, logic)                                                                                                                |
| 4   | **Step-Back Prompting**         | Derives relevant mathematical principles first, then solves; used as primary call for all math questions                                                                                 |
| 5   | **Least-to-Most Decomposition** | Breaks multi-step word problems into sub-problems                                                                                                                                        |
| 6   | **Few-Shot Exemplars**          | Injects domain-specific in-context examples (science_mcq, true_false)                                                                                                                    |
| 7   | **Reflection / Self-Critique**  | Re-examines initial answer and corrects if needed (word_problem, science_mcq)                                                                                                            |
| 8   | **Self-Refine**                 | Checks and corrects a math answer when it looks wrong (e.g. expression instead of number); math only                                                                                     |
| 9   | **Debate**                      | Two independent solver agents answer the question; a judge picks the better answer if they disagree (logic, commonsense)                                                                 |
| 10  | **Answer Extraction**           | Post-processes raw LLM output to return a clean final answer                                                                                                                             |

## Recommended Reading

A curated list of foundational papers and talks on LLM reasoning and agents:

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Rich Sutton
- [Learning to Self-Improve & Reason with LLMs](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf) - Berkeley RDI
- [LLM Reasoning: Key Ideas and Limitations](https://rdi.berkeley.edu/llm-agents/assets/llm-reasoning.pdf) - Berkeley RDI
- [Reasoning and Agents](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf) - Natural Language Processing with Deep Learning CS224N/Ling284
- [Inference-Time Techniques for LLM Reasoning](https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf) - Berkeley RDI

## Strategy per Domain

| Domain                  | Strategy                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| `math`                  | Step-Back → Self-Consistency (3 samples) → Self-Refine if needed → CoT fallback |
| `word_problem`          | Decomposition + CoT in parallel → Reflection if they disagree                   |
| `reading_comprehension` | Single tight extraction prompt (1 call)                                         |
| `science_mcq`           | Few-Shot → CoT → Reflection for letter extraction                               |
| `logic`                 | Debate (2 solvers + judge) → Self-Consistency (3 samples) → CoT fallback        |
| `true_false`            | Few-Shot (1 call)                                                               |
| `commonsense`           | CoT → Debate fallback for option-style questions → Verify + Reflection          |

## LLM Call Budget

- Math: 1 or 5 calls (1 step-back -> if needed, 3 self-consistency -> self-refine -> CoT)
- Word problem: ~2-3 calls (1 decompose + 1 CoT -> 1 reflect only if disagree)
- Reading comprehension: 1 call (single extract prompt)
- Science MCQ: ~1–3 calls (1 few shot -> 1 CoT if no letter -> 1 reflect if still no letter)
- Logic: ~2-6 calls (1 debate: 1 solver A + 1 solver B + 1 judge -> 3 self-consistency -> CoT fallback)
- True/False: ~1 call (few shot only)
- Commonsense: ~1–4 calls (1 CoT -> debate: 1 solver A + 1 solver B + 1 judge for option-style -> 1 reflect)

All well within the 20-call limit.

## How to Run the Evaluation

Steps to evaluate your agent on the dev questions:

```bash

# Run the evaluation on the dev questions
python generate_answer_template.py
```

This will:

1. Load the 6208 dev questions
2. Run your agent and generate answers

Then, you can see how your agent performs.

## Files

```
agent/llm.py                          # Main agent implementation
generate_answer_template.py           # Submission runner
eval/run_eval.py                      # Dev set evaluation script
requirements.txt                      # Python dependencies
cse476_final_project_dev_data.json    # Dev data
cse_476_final_project_test_data.json  # Test data (no answers)
cse_476_final_project_answers.json    # Output answers (generated)
```
