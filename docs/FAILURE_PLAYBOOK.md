# Failure Playbook

## How to Use This Document

After each eval run, look at the failures and match them to a pattern below.
Each pattern has a diagnosis and a specific fix.

---

## Pattern 1: Math — Model Gets Right Answer But Extractor Misses It

**Symptom**: 
```
expected='112'  got='m + n = 112'
expected='869'  got='Therefore sqrt(...) = 869.'
expected='5'    got='m + n = 5 where m=1, n=4'
```

**Diagnosis**: The model reasoned correctly but the extractor didn't find the number.

**Fix**: Add these patterns to `extractor.py`:
- `m\s*\+\s*n\s*=\s*(\d+)` → extract the number
- `=\s*(\d+)\s*\.?\s*$` → last equals sign before period
- `Therefore.*?(\d+)` → number after "therefore"

---

## Pattern 2: Math — Model Gives Wrong Answer (Reasoning Error)

**Symptom**:
```
expected='112'  got='56'
expected='164'  got='200'
```

**Diagnosis**: The model made a calculation or reasoning error.

**Fix options** (in order of preference):
1. Trigger self-refine: ask model to check its work
2. Improve the step-back prompt to be more specific about the method
3. Add a hint in the prompt about the type of problem (e.g., "this is a combinatorics problem")

---

## Pattern 3: Math — Model Gives Partial Answer

**Symptom**:
```
expected='98'   got='a + b + c'  (symbolic, not numeric)
expected='36'   got='m + n + p = ...'  (expression not evaluated)
```

**Diagnosis**: Model set up the problem but didn't compute the final number.

**Fix**: Add to the prompt: "Compute the final numeric value. Do not leave the answer as an expression."

---

## Pattern 4: Word Problem — Wrong Number Extracted

**Symptom**:
```
expected='7'    got='60'   (extracted the total instead of the answer)
expected='408'  got='3'    (extracted a ratio instead of the total)
```

**Diagnosis**: Extractor grabbed the wrong number from the response.

**Fix**: 
- Look for "Final answer:" prefix first
- Only fall back to "last number" if no explicit marker found
- Add to prompt: "State only the final answer number, nothing else, after 'Final answer:'"

---

## Pattern 5: Word Problem — Multilingual Question Fails

**Symptom**:
```
Question in Spanish/Chinese → empty or wrong answer
```

**Diagnosis**: Model may not handle the language well, or the prompt is English-only.

**Fix**: 
- Detect non-ASCII characters in question
- Add to prompt: "The question may be in another language. Understand it and answer in the same language or with just a number."
- The model (Qwen) handles Chinese well natively

---

## Pattern 6: MCQ — Answer Is Full Option Text Instead of Letter

**Symptom**:
```
expected='B'    got='B. chemical energy into radiant energy'
expected='C'    got='(C) The Lion King'
```

**Diagnosis**: Extractor not stripping the option text.

**Fix**: In extractor for MCQ domain:
- Pattern: `^([A-D])[\.\):]` → extract just the letter
- Pattern: `\(([A-D])\)` → extract letter from parentheses

---

## Pattern 7: Reading Comprehension — Answer Too Long

**Symptom**:
```
expected='1939'  got='The Tower Theatre was built in 1939 and is located at...'
expected='Warner Bros.'  got='It was released by Warner Bros. Records on...'
```

**Diagnosis**: Model gives a full sentence instead of the key span.

**Fix**: Tighten the prompt:
- "Answer with the shortest possible phrase that directly answers the question."
- "Do not include surrounding context. Just the answer."

---

## Pattern 8: True/False — Answer Has Explanation

**Symptom**:
```
expected='No'   got='No, because Bombyx mori does not have a monopoly since spiders also produce silk.'
```

**Diagnosis**: Model gives correct answer but with explanation.

**Fix**: 
- Extractor: check first word of response for Yes/No
- Prompt: "Answer with exactly one word: Yes or No."

---

## Pattern 9: Logic — Wrong Option Selected

**Symptom**:
```
expected='(C)'  got='(A)'
```

**Diagnosis**: Model traced the logic incorrectly.

**Fix**:
- Add explicit state tracking to the prompt: "Keep track of who has what after each swap."
- Format: "After step 1: Alice has X, Bob has Y, Claire has Z. After step 2: ..."

---

## Pattern 10: Domain Misclassification

**Symptom**: A math problem gets routed to word_problem strategy, or a logic puzzle gets routed to commonsense.

**Diagnosis**: The heuristic router made the wrong call.

**Fix**: 
- Look at what signal the router used
- Add a more specific heuristic for that question type
- Check: does the question have LaTeX? → math. Does it have "Options: (A)"? → logic.

---

## Pattern 11: Empty Response

**Symptom**:
```
expected='112'  got=''
```

**Diagnosis**: API call failed or returned empty.

**Fix**:
- Check API connectivity
- Add retry logic in `llm.py` (1 retry on empty response)
- Log the error for debugging

---

## Failure Triage Priority

When you have limited time, fix in this order:

1. **Extraction failures** (Pattern 1, 4, 6, 7, 8) — easy wins, no new LLM calls needed
2. **Prompt improvements** (Pattern 3, 9) — small prompt changes, big impact
3. **Router fixes** (Pattern 10) — affects all questions of that type
4. **Self-refine triggers** (Pattern 2) — adds 1 call but improves hard cases
5. **Multilingual** (Pattern 5) — affects a subset of word problems
6. **Empty responses** (Pattern 11) — infrastructure issue, fix last
