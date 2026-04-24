#!/usr/bin/env python3
"""
Evaluation runner — measures agent accuracy on dev data.

Usage:
  python eval/run_eval.py                        # all 1000 dev questions
  python eval/run_eval.py --n 50                 # first 50
  python eval/run_eval.py --n 50 --start 100     # questions 100-149
  python eval/run_eval.py --domain math          # only math questions
  python eval/run_eval.py --n 50 --no-llm-judge  # skip LLM judge (faster)
  python eval/run_eval.py --sample fixed         # always same 50 questions
  python eval/run_eval.py --sample random        # random dev sample matching routed test mix
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import agent_loop
from agent import llm
from agent.router import detect_domain
from eval.grader import grade

DEV_PATH     = Path("cse476_final_project_dev_data.json")
TEST_PATH    = Path("cse_476_final_project_test_data.json")
RESULTS_DIR  = Path("eval/results")
FIXED_SAMPLE = Path("eval/fixed_sample_indices.json")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dev_data(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_fixed_sample(dev_data: list, n: int = 50) -> list[int]:
    """Return fixed indices covering all domains proportionally."""
    if FIXED_SAMPLE.exists():
        with FIXED_SAMPLE.open() as f:
            indices = json.load(f)
        return _interleave_indices_by_domain(dev_data, indices)[:n]

    # Build a balanced sample
    by_domain: dict = defaultdict(list)
    for i, item in enumerate(dev_data):
        domain = item.get("domain", detect_domain(item["input"]))
        by_domain[domain].append(i)

    # Proportional allocation
    total = n
    domain_counts = {d: len(v) for d, v in by_domain.items()}
    total_items = sum(domain_counts.values())

    indices = []
    for domain, items in by_domain.items():
        alloc = max(1, round(len(items) / total_items * total))
        indices.extend(items[:alloc])

    indices = _interleave_indices_by_domain(dev_data, sorted(indices[:n]))

    # Save for future runs
    with FIXED_SAMPLE.open("w") as f:
        json.dump(indices, f)

    return indices


def _interleave_indices_by_domain(dev_data: list, indices: list[int]) -> list[int]:
    """Order a fixed sample so early console output covers all domains."""
    by_domain: dict = defaultdict(list)
    domain_order = []

    for idx in indices:
        domain = dev_data[idx].get("domain", detect_domain(dev_data[idx]["input"]))
        if domain not in by_domain:
            domain_order.append(domain)
        by_domain[domain].append(idx)

    interleaved = []
    while any(by_domain.values()):
        for domain in domain_order:
            if by_domain[domain]:
                interleaved.append(by_domain[domain].pop(0))

    return interleaved


def get_testmix_sample(
    dev_data: list,
    test_path: Path,
    n: int = 50,
) -> tuple[list[int], Counter, Counter, list[str]]:
    """
    Return random dev indices whose routed-domain mix approximates the test set.

    The test file has no labels, so we estimate its mix with the same router the
    agent uses. If a routed test domain has no matching dev examples, we report
    it and redistribute those slots across available routed domains.
    """
    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    target_mix = Counter(detect_domain(item["input"]) for item in test_data)

    dev_by_route: dict = defaultdict(list)
    for idx, item in enumerate(dev_data):
        routed = detect_domain(item["input"])
        dev_by_route[routed].append(idx)

    missing_domains = [
        domain for domain in sorted(target_mix)
        if domain not in dev_by_route
    ]
    available_weights = Counter({
        domain: count
        for domain, count in target_mix.items()
        if domain in dev_by_route
    })
    capacities = {
        domain: len(dev_by_route[domain])
        for domain in available_weights
    }
    allocations = _allocate_proportionally(n, available_weights, capacities)

    indices = []
    for domain, count in allocations.items():
        pool = list(dev_by_route[domain])
        random.shuffle(pool)
        indices.extend(pool[:count])

    random.shuffle(indices)
    selected_mix = Counter(detect_domain(dev_data[idx]["input"]) for idx in indices)
    return indices, target_mix, selected_mix, missing_domains


def _allocate_proportionally(
    n: int,
    weights: Counter,
    capacities: dict[str, int],
) -> Counter:
    """Largest-remainder allocation with per-domain capacity limits."""
    total_weight = sum(weights.values())
    if n <= 0 or total_weight <= 0:
        return Counter()

    raw = {
        domain: n * weight / total_weight
        for domain, weight in weights.items()
    }
    allocation = Counter({
        domain: min(int(value), capacities[domain])
        for domain, value in raw.items()
    })

    remaining = n - sum(allocation.values())
    while remaining > 0:
        candidates = [
            domain for domain in weights
            if allocation[domain] < capacities[domain]
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda domain: (
                raw[domain] - int(raw[domain]),
                weights[domain],
            ),
            reverse=True,
        )
        for domain in candidates:
            if remaining <= 0:
                break
            allocation[domain] += 1
            remaining -= 1

    return allocation


def _format_counter(counter: Counter, total: int | None = None) -> str:
    total = total or sum(counter.values()) or 1
    return ", ".join(
        f"{domain}: {count} ({count / total * 100:.1f}%)"
        for domain, count in counter.most_common()
    )


def run_eval(
    dev_data: list,
    indices: list[int],
    use_llm_judge: bool = True,
    verbose: bool = True,
    print_questions: bool = False,
) -> dict:
    """Run evaluation on specified indices. Returns results dict."""
    results = []
    domain_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    total = len(indices)
    for rank, idx in enumerate(indices, 1):
        item = dev_data[idx]
        question = item["input"]
        expected = item["output"]
        domain   = item.get("domain", detect_domain(question))
        routed_domain = detect_domain(question)

        t0 = time.time()
        try:
            prediction = agent_loop(question)
        except Exception as e:
            prediction = ""
            if verbose:
                print(f"  [ERROR] idx={idx}: {e}", file=sys.stderr)

        elapsed = time.time() - t0

        is_correct = grade(question, prediction, expected, use_llm_judge=use_llm_judge, domain=domain)

        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1

        result = {
            "index":      idx,
            "domain":     domain,
            "routed_domain": routed_domain,
            "question":   question[:100],
            "expected":   expected,
            "prediction": prediction,
            "correct":    is_correct,
            "elapsed":    round(elapsed, 2),
        }
        results.append(result)

        if verbose:
            mark = "✅" if is_correct else "❌"
            running_correct = sum(1 for r in results if r["correct"])
            route_note = f" -> {routed_domain}" if routed_domain != domain else ""
            if print_questions:
                print(f"\n[{rank:3d}/{total}] Q: {question}")
            print(
                f"{mark} [{rank:3d}/{total}] [{domain:22s}{route_note}] "
                f"exp={expected!r:12s} got={prediction!r:25s} "
                f"({elapsed:.1f}s)"
            )

    # Summary
    total_correct = sum(1 for r in results if r["correct"])
    overall_pct = total_correct / len(results) * 100 if results else 0

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "total":        len(results),
        "correct":      total_correct,
        "accuracy":     round(overall_pct, 1),
        "by_domain":    {
            d: {
                "correct":  s["correct"],
                "total":    s["total"],
                "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0,
            }
            for d, s in sorted(domain_stats.items())
        },
        "results": results,
    }

    return summary


def print_summary(summary: dict) -> None:
    print()
    print("=" * 65)
    print(f"OVERALL: {summary['correct']}/{summary['total']} = {summary['accuracy']}%")
    print("=" * 65)
    print(f"{'Domain':<28} {'Correct':>7} {'Total':>6} {'Accuracy':>9}")
    print("-" * 55)
    for domain, stats in summary["by_domain"].items():
        bar = "█" * int(stats["accuracy"] / 5)
        print(
            f"  {domain:<26} {stats['correct']:>7} {stats['total']:>6} "
            f"{stats['accuracy']:>8.1f}%  {bar}"
        )
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the reasoning agent on dev data")
    parser.add_argument("--n",           type=int,  default=None,  help="Number of questions")
    parser.add_argument("--start",       type=int,  default=0,     help="Start index")
    parser.add_argument("--domain",      type=str,  default=None,  help="Filter by domain")
    parser.add_argument("--sample",      type=str,  default=None,  choices=["fixed", "random"], help="'fixed' for stable sample, 'random' for random sample matching routed test mix")
    parser.add_argument("--no-llm-judge",action="store_true",      help="Skip LLM judge (faster, less accurate grading)")
    parser.add_argument("--quiet",       action="store_true",      help="Suppress per-question output")
    parser.add_argument("--printquestions",action="store_true",    help="Print the full question text in the output")
    args = parser.parse_args()

    print(f"Loading dev data from {DEV_PATH}...")
    dev_data = load_dev_data(DEV_PATH)
    print(f"Loaded {len(dev_data)} questions.")

    # Determine which questions to evaluate
    if args.sample == "fixed":
        n = args.n if args.n is not None else 50
        indices = get_fixed_sample(dev_data, n)
        print(f"Using fixed sample of {len(indices)} questions.")
    elif args.sample == "random":
        n = args.n if args.n is not None else 50
        indices, target_mix, selected_mix, missing_domains = get_testmix_sample(
            dev_data,
            TEST_PATH,
            n,
        )
        print(f"Using random test-mix sample of {len(indices)} questions.")
        print(f"Routed test mix: {_format_counter(target_mix)}")
        print(f"Selected routed mix: {_format_counter(selected_mix)}")
        if missing_domains:
            print(
                "No matching dev examples for routed test domains: "
                + ", ".join(missing_domains)
                + ". Redistributed those slots across available domains."
            )
    elif args.domain:
        all_indices = [
            i for i, item in enumerate(dev_data)
            if item.get("domain", detect_domain(item["input"])) == args.domain
        ]
        end = args.start + args.n if args.n else len(all_indices)
        indices = all_indices[args.start:end]
        print(f"Filtered to {len(indices)} '{args.domain}' questions.")
    else:
        end = args.start + args.n if args.n else len(dev_data)
        indices = list(range(args.start, min(end, len(dev_data))))
        print(f"Evaluating questions {args.start}–{args.start + len(indices) - 1}.")

    use_llm_judge = not args.no_llm_judge
    print(f"LLM judge: {'enabled' if use_llm_judge else 'disabled'}")
    print(f"LLM API key: {'found' if llm.is_configured() else 'MISSING'}")
    print(f"LLM API base: {llm.API_BASE}")
    print(f"LLM model: {llm.MODEL}")
    if not llm.is_configured():
        print(
            "WARNING: API_KEY/OPENAI_API_KEY is missing. "
            "Non-planning strategies will return empty predictions; "
            "set it in your shell or repo-root .env."
        )
    print()

    summary = run_eval(dev_data, indices, use_llm_judge=use_llm_judge, verbose=not args.quiet, print_questions=args.printquestions)
    print_summary(summary)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also save as latest
    latest_path = RESULTS_DIR / "latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
