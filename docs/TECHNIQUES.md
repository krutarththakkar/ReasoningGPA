# Inference-Time Techniques — Deep Dive

## Overview

Each technique is a distinct way of getting better answers from the LLM at inference time,
without changing the model weights. We implement 8 techniques, each in its own module.

---

## Technique 1: Chain-of-Thought (CoT)

**File**: `agent/techniques/cot.py`

**What it is**: Ask the model to reason step by step before giving the final answer.

**Why it works**: Forces the model to "show its work", which reduces errors on multi-step problems.

**Prompt pattern**:
```
System: You are a careful problem solver. Think step by step.
        At the end, state your final answer on its own line prefixed with "Final answer:"

User: [question]
      
      Solve step by step:
```

**When to use**: All domains as the base technique.

**Tokens needed**: 
- Math: 1200 (needs room to show work)
- Word problems: 600
- Everything else: 300

**Key implementation detail**: The system prompt must explicitly say "Final answer:" — this is what the extractor looks for.

---

## Technique 2: Domain-Aware Routing

**File**: `agent/router.py`

**What it is**: Classify the question type before choosing a strategy.

**Why it works**: Different domains need completely different prompting strategies. A math prompt is wrong for reading comprehension.

**How we detect domains (heuristics, no LLM call)**:

| Signal | Domain |
|--------|--------|
| LaTeX patterns (`\frac`, `\sqrt`, `\triangle`, `$`) | math |
| "Options: (A)" or "(A) ... (B) ... (C)" | logic or science_mcq |
| "A. ... B. ... C. ... D." with science keywords | science_mcq |
| "Facts:" or "Is X true?" | true_false |
| Long context passage (>600 chars) + question at end | reading_comprehension |
| Numbers + "how many/much/total/earn/save" | word_problem |
| "most likely", "similar to", "plausible" | commonsense |

**Fallback**: If no heuristic matches, default to `commonsense`.

**Key constraint**: Zero LLM calls. This must be instant.

---

## Technique 3: Step-Back Prompting

**File**: `agent/techniques/step_back.py`

**What it is**: Before solving, ask the model to identify the relevant mathematical principles and formulas. Then use those to solve.

**Why it works**: Forces the model to retrieve relevant knowledge before applying it. Reduces "jumping to wrong method" errors on hard math.

**Prompt pattern**:
```
Step 1: What mathematical concepts, theorems, or formulas are relevant to this problem?
        List them briefly.

Step 2: Using those concepts, solve the problem step by step.

Step 3: State the final answer as a number.

Problem: [question]
```

**When to use**: Hard math problems (AIME-style, LaTeX-heavy).

**Tokens needed**: 1200

**Call count**: 1 (replaces CoT for hard math, doesn't add to it)

---

## Technique 4: Least-to-Most Decomposition

**File**: `agent/techniques/decompose.py`

**What it is**: Break a complex problem into simpler sub-problems. Solve each sub-problem in order, using previous answers as context.

**Why it works**: Multi-step word problems often fail because the model tries to do everything at once. Decomposition forces sequential reasoning.

**Prompt pattern**:
```
Break this problem into smaller steps:

Problem: [question]

Step 1: What is the first quantity we need to find?
Step 2: What is the next quantity?
...
Final: Combine the results to get the answer.

Work through each step:
```

**When to use**: Word problems, especially multi-step ones.

**Tokens needed**: 600

**Call count**: 1 (single call with decomposition prompt)

---

## Technique 5: Few-Shot Exemplars

**File**: `agent/techniques/few_shot.py`

**What it is**: Include 1-2 solved examples in the prompt before the actual question.

**Why it works**: Shows the model the expected format and reasoning style. Especially useful for MCQ (shows "answer with just the letter") and true/false (shows "answer Yes or No").

**Examples per domain**:

### Science MCQ example:
```
Question: Which process converts sunlight into chemical energy?
A. respiration  B. photosynthesis  C. digestion  D. fermentation
Answer: B

Question: [actual question]
Answer:
```

### True/False example:
```
Facts: Dogs are mammals. Mammals are warm-blooded.
Question: Are dogs warm-blooded?
Answer: Yes

Facts: [actual facts]
Question: [actual question]
Answer:
```

**When to use**: MCQ, true/false, commonsense.

**Tokens needed**: 300 (examples are short)

**Call count**: 0 extra (examples are in the prompt, not separate calls)

---

## Technique 6: Self-Refine

**File**: `agent/techniques/self_refine.py`

**What it is**: After getting an initial answer, ask the model to review it and correct any mistakes.

**Why it works**: The model can often catch its own errors when explicitly asked to check. Especially useful for math where a calculation might be off.

**Prompt pattern**:
```
System: You are a careful reviewer. Check if the answer is correct.
        If wrong, provide the correct answer. State "Final answer: X" at the end.

User: Question: [question]
     
     My answer was: [initial_answer]
     
     Please check this answer carefully. Is it correct?
     If not, what is the correct answer? Show your work.
```

**When to use**: Math only, and only when:
- The initial answer is non-numeric (extraction failed)
- The initial answer seems implausible (negative when should be positive, etc.)
- We have budget remaining (< 3 calls used so far)

**Tokens needed**: 800

**Call count**: +1 (only triggered conditionally)

---

## Technique 7: Answer Extraction + Normalization

**File**: `agent/extractor.py`

**What it is**: Post-process the raw LLM output to extract a clean final answer.

**Why it matters**: The model often gives correct answers buried in explanation. The extractor finds them.

**Patterns to handle** (in priority order):

1. `Final answer: 112` → `112`
2. `\boxed{112}` → `112`
3. `**112**` (bold markdown) → `112`
4. `Therefore, the answer is 112.` → `112`
5. `m + n = 112` → `112`
6. `The answer is B.` → `B`
7. `(C) blue ball` → `C` or `blue ball` depending on domain
8. `Yes, because...` → `Yes`
9. Last short line (< 50 chars) → use as answer

**Domain-specific extraction**:
- Math: extract the last number or expression
- MCQ: extract the letter (A/B/C/D)
- True/False: extract Yes/No
- Reading comp: extract the relevant span
- Logic: extract the option or description

**Call count**: 0 (pure string processing)

---

## Technique 8: LLM-as-Judge (Self-Evaluation)

**File**: `agent/techniques/self_eval.py`

**What it is**: Use the LLM itself to verify whether an answer is correct.

**Why it works**: The model can often judge correctness even when it can't solve the problem directly.

**Prompt pattern**:
```
System: You are a strict grader. Reply with exactly True or False.

User: QUESTION: [question]
     ANSWER: [predicted_answer]
     
     Is this answer correct? Reply True or False.
```

**When to use**: 
- In the eval script (to grade our own outputs during development)
- In the agent for math: after self-refine, verify the refined answer
- Only when we have budget remaining

**Call count**: +1 (used sparingly)

**Important**: This is also how the tutorial's grader works. So our self-eval mimics the actual grader.

---

## Technique Interaction Map

```
Question
    │
    ▼
[Router] ──────────────────────────────────────────────────────────
    │                                                              │
    ▼ math                                                         ▼ others
[Step-Back CoT]                                            [CoT or Few-Shot]
    │                                                              │
    ▼                                                              ▼
[Extractor]                                                  [Extractor]
    │                                                              │
    ├── answer looks good ──────────────────────────────────► return
    │
    ▼ answer looks wrong
[Self-Refine]
    │
    ▼
[Extractor]
    │
    ▼
[Self-Eval] (optional, if budget allows)
    │
    ▼
return
```

---

## What Counts as "Distinct" Techniques

The project requires 8 distinct techniques. Here's how we justify each:

1. **CoT** — prompting strategy (step-by-step reasoning)
2. **Domain-Aware Routing** — meta-strategy (choosing the right approach)
3. **Step-Back Prompting** — prompting strategy (principles before solving)
4. **Least-to-Most Decomposition** — prompting strategy (sub-problem breakdown)
5. **Few-Shot Exemplars** — prompting strategy (in-context learning)
6. **Self-Refine** — iterative refinement (multi-turn correction)
7. **Answer Extraction + Normalization** — output processing (post-generation)
8. **LLM-as-Judge** — verification (using model to evaluate model output)

Each is conceptually distinct and implemented in a separate module.
