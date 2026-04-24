#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import agent_loop
from agent.router import detect_domain
from eval.grader import grade

DEV_PATH = Path("cse476_final_project_dev_data.json")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_dev_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_fixed_sample(dev_data, n=50):
    # Return first n indices for simplicity
    return list(range(min(n, len(dev_data))))

def run_eval(dev_data, indices, use_llm_judge=True, verbose=True):
    results = []
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for rank, idx in enumerate(indices, 1):
        item = dev_data[idx]
        question = item["input"]
        expected = item["output"]
        domain = item.get("domain", detect_domain(question))
        t0 = time.time()
        try:
            prediction = agent_loop(question)
        except:
            prediction = ""
            if verbose:
                print(f"Error on index {idx}")
        elapsed = time.time() - t0
        correct = grade(question, prediction, expected, use_llm_judge=use_llm_judge)
        domain_stats[domain]["total"] += 1
        if correct:
            domain_stats[domain]["correct"] += 1
        results.append({
            "index": idx,
            "domain": domain,
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "correct": correct,
            "elapsed": round(elapsed, 2),
        })
        if verbose:
            mark = "✅" if correct else "❌"
            print(f"{mark} Index {idx}: {question[:50]} ... Predicted: {prediction}")

    total_correct = sum(r["correct"] for r in results)
    accuracy = total_correct / len(results) * 100 if results else 0
    return {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "correct": total_correct,
        "accuracy": round(accuracy, 1),
        "by_domain": domain_stats,
        "results": results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dev_data = load_dev_data(DEV_PATH)
    if args.sample == "fixed":
        indices = get_fixed_sample(dev_data, args.n or 50)
    elif args.domain:
        indices = [i for i, item in enumerate(dev_data) if item.get("domain", detect_domain(item["input"])) == args.domain]
        indices = indices[args.start: (args.start + args.n) if args.n else None]
    else:
        end = args.start + (args.n or len(dev_data))
        indices = list(range(args.start, min(end, len(dev_data))))
    use_llm_judge = not args.no_llm_judge

    summary = run_eval(dev_data, indices, use_llm_judge=use_llm_judge, verbose=not args.quiet)
    print(f"Accuracy: {summary['accuracy']}%")
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with out_path.open("w") as f:
        json.dump(summary, f)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
