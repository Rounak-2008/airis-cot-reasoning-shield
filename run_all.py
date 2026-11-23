#!/usr/bin/env python3
"""
AIRIS Research Hackathon (Nov 22-24, 2026)
Master Experiment Runner for:
- Paper #2: Unfaithful Explanations in CoT (Turpin et al., NeurIPS 2023) -> BBQ & BBH
- Paper #5: Towards Understanding Sycophancy (Sharma et al., ICLR 2024) -> Answer QA & Are You Sure?

Runs with one command as required by the hackathon submission rubric.
Loads API credentials from local .env (or prompts once and saves locally).
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


def ensure_api_key(cli_key: str = None) -> str:
    """Loads Hackathon API key from CLI, environment, or .env.
    If missing, prompts once and persists to .env for future runs."""
    env_path = os.path.join(BASE_DIR, ".env")

    # 1. Check CLI argument
    if cli_key:
        os.environ["HACKATHON_API_KEY"] = cli_key.strip()
        return cli_key.strip()

    # 2. Check existing environment
    if os.environ.get("HACKATHON_API_KEY"):
        return os.environ["HACKATHON_API_KEY"].strip()

    # 3. Check .env file
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "HACKATHON_API_KEY":
                        val = v.strip().strip("'\"")
                        if val and val != "your_hackathon_api_key_here":
                            os.environ["HACKATHON_API_KEY"] = val
                            return val

    # 4. Prompt user once and save forever
    print("\n" + "=" * 65)
    print("Hackathon API Key not detected in environment or .env!")
    print("Please enter your Hackathon API Key.")
    print("It will be securely saved locally to .env (git-ignored) for future runs.")
    print("=" * 65)
    try:
        key_input = input("Enter HACKATHON_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        key_input = ""

    if key_input:
        os.environ["HACKATHON_API_KEY"] = key_input
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\nHACKATHON_API_KEY={key_input}\n")
        print(f"API Key saved to {env_path}\n")
        return key_input
    else:
        print("Warning: Proceeding without API key (may cause authorization errors).\n")
        return ""


def run_benchmark(name: str, limit: int = None):
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

    print(f"\n{'='*70}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    res = subprocess.run(cmd, env=env)
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
    parser.add_argument("--api-key", type=str, default=None, help="Hackathon API key (saves to .env)")
    args = parser.parse_args()

    api_key = ensure_api_key(args.api_key)

    print(f"=== AIRIS Research Hackathon Benchmark Suite ===")
    print(f"Target Models: Qwen3-30B-A3B (Sparse MoE) vs Qwen3-32B (Dense)")
    print(f"Mode: OFFICIAL HACKATHON CLUSTER API (http://10.7.25.115:30400/v1)")
    print(f"API Key: {'configured (' + api_key[:6] + '...' + api_key[-4:] + ')' if api_key else 'MISSING'}")

    if args.benchmark == "all":
        order = ["bbq", "answer", "are_you_sure", "bbh", "feedback"]
        for b in order:
            print(f"\n>>> Running Benchmark: {b.upper()} <<<")
            run_benchmark(b, limit=args.limit)
    else:
        run_benchmark(args.benchmark, limit=args.limit)

    print("\nBenchmark run complete! Logs are saved in ./logs/")


if __name__ == "__main__":
    main()
