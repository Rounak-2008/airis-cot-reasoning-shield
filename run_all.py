#!/usr/bin/env python3
"""
AIRIS Research Hackathon (Nov 22-24, 2026)
Master Experiment Runner for:
- Paper #2: Unfaithful Explanations in CoT (Turpin et al., NeurIPS 2023) -> BBQ & BBH
- Paper #5: Towards Understanding Sycophancy (Sharma et al., ICLR 2024) -> Answer QA & Are You Sure?

Runs with one command as required by the hackathon submission rubric.
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = {
    "bbq": ("run_bbq.py", ["--limit", "10"]),
    "bbh": ("dry_run.py", ["--limit", "15"]),
    "answer": ("run_paper5_answer.py", ["--limit", "50"]),
    "are_you_sure": ("run_paper5_are_you_sure.py", ["--limit", "50"]),
    "feedback": ("run_paper5_feedback.py", ["--limit", "50"]),
}


def run_benchmark(name: str, limit: int = None, use_local: bool = False):
    if name not in BENCHMARKS:
        print(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
        return False

    script_name, default_args = BENCHMARKS[name]
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False

    cmd = [sys.executable, script_path]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    else:
        cmd.extend(default_args)

    if use_local:
        cmd.append("--local")

    print(f"\n{'='*70}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    res = subprocess.run(cmd)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AIRIS Hackathon Master Runner")
    parser.add_argument(
        "--benchmark",
        choices=["all", "bbq", "bbh", "answer", "are_you_sure", "feedback"],
        default="bbq",
        help="Benchmark to execute (default: bbq)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Items per task/file (override default)")
    parser.add_argument("--local", action="store_true", help="Run exclusively against local Ollama instance")
    args = parser.parse_args()

    print(f"=== AIRIS Research Hackathon Benchmark Suite ===")
    print(f"Target Models: Qwen3-30B-A3B (Sparse MoE) vs Qwen3-32B (Dense)")
    print(f"Mode: {'LOCAL (Ollama)' if args.local else 'REMOTE CLUSTER API'}")

    if args.benchmark == "all":
        order = ["bbq", "answer", "are_you_sure", "bbh", "feedback"]
        for b in order:
            print(f"\n>>> Running Benchmark: {b.upper()} <<<")
            run_benchmark(b, limit=args.limit, use_local=args.local)
    else:
        run_benchmark(args.benchmark, limit=args.limit, use_local=args.local)

    print("\nBenchmark run complete! Logs are saved in ./logs/")


if __name__ == "__main__":
    main()
