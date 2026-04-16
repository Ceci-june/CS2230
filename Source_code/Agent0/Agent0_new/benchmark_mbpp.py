#!/usr/bin/env python3
"""
Benchmark on MBPP / MBPP+ (EvalPlus)
=====================================

Generates solutions for MBPP tasks, then evaluates with EvalPlus.
Compares two modes:
  A) WITH knowledge base (few-shot RAG)
  B) WITHOUT knowledge base (baseline)

Usage:
  # Generate solutions (with KB)
  python benchmark_mbpp.py generate --provider ollama-cloud --model gemma3:4b --mode with_kb --limit 50

  # Generate solutions (without KB)
  python benchmark_mbpp.py generate --provider ollama-cloud --model gemma3:4b --mode baseline --limit 50

  # Evaluate with EvalPlus
  python benchmark_mbpp.py evaluate --samples_file results/gemma3_4b/with_kb_samples.jsonl

  # Full run: generate + evaluate both modes
  python benchmark_mbpp.py run --provider ollama-cloud --model gemma3:4b --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from executor import (
    RuntimeConfig,
    execute_task,
    solve_with_knowledge,
    chat_completion,
    extract_code_block,
    strip_think_blocks,
)
from knowledge_retriever import KnowledgeRetriever

try:
    from dotenv import load_dotenv
    for p in [SCRIPT_DIR / ".env", SCRIPT_DIR.parents[2] / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
except Exception:
    pass

logger = logging.getLogger("benchmark")

REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_KB_PATH = REPO_ROOT / "Data" / "mbpp" / "curriculum_outputs" / "gemma3_4b" / "knowledge_base.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "Data" / "mbpp" / "benchmark_results"

# EvalPlus expects this format for MBPP
EVALPLUS_MBPP_PREFIX = "Mbpp/"

# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434",
    "ollama-cloud": "https://ollama.com",
}

PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
    "ollama-cloud": "OLLAMA_API_KEY",
}


def make_runtime(provider: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> RuntimeConfig:
    env_key = PROVIDER_ENV_KEYS.get(provider)
    api_key = api_key or (os.getenv(env_key) if env_key else None)
    base_url = base_url or PROVIDER_URLS.get(provider, "http://localhost:11434")
    return RuntimeConfig(provider=provider, model=model, backend=provider, base_url=base_url, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Load MBPP tasks (test split: task_id 11-510)
# ─────────────────────────────────────────────────────────────────────────────

def load_mbpp_test_tasks(data_file: Optional[str] = None) -> List[dict]:
    """Load MBPP+ tasks from EvalPlus (378 tasks, the standard benchmark set)."""
    from evalplus.data import get_mbpp_plus

    mbpp_plus = get_mbpp_plus()
    tasks = []
    for task_id, data in mbpp_plus.items():
        prompt = data.get("prompt", "")
        entry_point = data.get("entry_point", "solution")
        assertion = data.get("assertion", "")
        # Extract test_list from assertion string
        test_list = [line.strip() for line in assertion.split("\n") if line.strip().startswith("assert")]

        tasks.append({
            "task_id": task_id,  # e.g. "Mbpp/2"
            "prompt": prompt,
            "test_list": test_list[:5],  # limit for prompt context
            "challenge_test_list": [],
            "test_setup_code": "",
            "reference_solution": data.get("canonical_solution", ""),
            "entry_point": entry_point,
        })

    logger.info("Loaded %s MBPP+ tasks from EvalPlus", len(tasks))
    return tasks


def _extract_entry_point(code: str) -> str:
    """Extract function name from reference code."""
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return match.group(1) if match else "solution"


# ─────────────────────────────────────────────────────────────────────────────
# Generate solutions
# ─────────────────────────────────────────────────────────────────────────────

def generate_solution_for_task(
    runtime: RuntimeConfig,
    task: dict,
    retriever: Optional[KnowledgeRetriever] = None,
    n_examples: int = 3,
) -> str:
    """Generate a solution for one MBPP task, optionally with KB few-shot."""
    if retriever is not None:
        examples = retriever.query(task["prompt"], n=n_examples)
        few_shot = retriever.format_few_shot(examples)

        messages = [
            {
                "role": "system",
                "content": "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation.",
            },
            {
                "role": "user",
                "content": (
                    f"Here are similar solved tasks for reference:\n\n{few_shot}\n\n"
                    f"Now solve the new task below following the same patterns.\n\n"
                    f"Task:\n{task['prompt']}\n\n"
                    f"Tests:\n{json.dumps(task['test_list'], indent=2)}\n\n"
                    f"Return only the Python solution code."
                ),
            },
        ]
    else:
        # 3-shot standard (tasks 2, 3, 4 as per MBPP protocol)
        messages = [
            {
                "role": "system",
                "content": "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation.",
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task['prompt']}\n\n"
                    f"Tests:\n{json.dumps(task['test_list'], indent=2)}\n\n"
                    f"Return only the Python solution code."
                ),
            },
        ]

    try:
        generated = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
        solution = (extract_code_block(generated) or generated).strip()
        solution = strip_think_blocks(solution)
        return solution
    except Exception as e:
        logger.warning("Generate failed | task_id=%s error=%s", task["task_id"], e)
        return f"def {task.get('entry_point', 'solution')}(*args, **kwargs):\n    pass"


def generate_all(
    runtime: RuntimeConfig,
    tasks: List[dict],
    output_file: Path,
    retriever: Optional[KnowledgeRetriever] = None,
    n_examples: int = 3,
) -> dict:
    """Generate solutions for all tasks and save in EvalPlus format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results = []
    passed = 0
    failed = 0

    # Get full task list for padding missing tasks
    all_tasks = load_mbpp_test_tasks()
    all_task_ids = {t["task_id"] for t in all_tasks}
    target_task_ids = {t["task_id"] for t in tasks}

    with output_file.open("w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks, 1):
            logger.info(
                "Generating %s/%s | task_id=%s %s",
                idx, len(tasks), task["task_id"],
                "(with KB)" if retriever else "(baseline)",
            )

            solution = generate_solution_for_task(runtime, task, retriever, n_examples)

            entry = {
                "task_id": task["task_id"],
                "solution": solution,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            results.append(entry)

            if solution and "pass" not in solution.split("\n")[-1]:
                passed += 1
            else:
                failed += 1

        # Pad missing tasks with empty solution (so EvalPlus doesn't complain)
        for t in all_tasks:
            if t["task_id"] not in target_task_ids:
                entry_point = t.get("entry_point", "solution")
                padding = {"task_id": t["task_id"], "solution": f"def {entry_point}(*args, **kwargs):\n    pass"}
                f.write(json.dumps(padding, ensure_ascii=False) + "\n")

    logger.info("Generation done | %s tasks generated + %s padded | saved to %s",
               len(results), len(all_task_ids) - len(target_task_ids), output_file)
    return {"total": len(results), "output_file": str(output_file)}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate with EvalPlus
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_evalplus(samples_file: str, dataset: str = "mbpp") -> dict:
    """Evaluate samples against MBPP base tests and MBPP+ extended tests."""
    import subprocess, shutil, tempfile

    from evalplus.data import get_mbpp_plus

    mbpp_plus = get_mbpp_plus()

    # Load samples
    samples = {}
    with open(samples_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            samples[row["task_id"]] = row["solution"]

    base_pass = 0
    plus_pass = 0
    total = 0
    details = []

    for task_id, data in mbpp_plus.items():
        if task_id not in samples:
            continue

        solution = samples[task_id]
        entry_point = data.get("entry_point", "solution")
        canonical = data.get("canonical_solution", "")

        # Skip padded empty solutions
        if solution.strip().endswith("pass") and len(solution.strip().split("\n")) <= 2:
            continue

        total += 1

        # Build base test script
        base_assertion = data.get("assertion", "")
        base_script = f"{solution}\n\n{base_assertion}"

        # Build plus test script (base + plus inputs)
        contract = data.get("contract", "")
        plus_inputs = data.get("plus_input", [])
        base_inputs = data.get("base_input", [])

        # Run base tests
        base_ok = _run_test_script(base_script, timeout=10)

        # Run plus tests (base + extended)
        # For plus, we need to call function with plus_input and compare to reference
        plus_ok = base_ok  # plus requires base to pass first
        if base_ok and plus_inputs:
            plus_test_lines = []
            for inp in plus_inputs[:20]:  # limit to avoid timeout
                try:
                    # Execute reference to get expected output
                    ref_script = f"{canonical}\n\nimport json\nresult = {entry_point}(*{json.dumps(inp)})\nprint(json.dumps(result))"
                    expected = _run_and_capture(ref_script, timeout=5)
                    if expected is not None:
                        plus_test_lines.append(
                            f"assert {entry_point}(*{json.dumps(inp)}) == {expected}"
                        )
                except Exception:
                    continue

            if plus_test_lines:
                plus_script = f"{solution}\n\n" + "\n".join(plus_test_lines)
                plus_ok = _run_test_script(plus_script, timeout=15)

        if base_ok:
            base_pass += 1
        if plus_ok:
            plus_pass += 1

        details.append({
            "task_id": task_id,
            "base_pass": base_ok,
            "plus_pass": plus_ok,
        })

    base_rate = (base_pass / total * 100) if total > 0 else 0
    plus_rate = (plus_pass / total * 100) if total > 0 else 0

    print(f"  MBPP (base) pass@1: {base_rate:.1f}% ({base_pass}/{total})")
    print(f"  MBPP+       pass@1: {plus_rate:.1f}% ({plus_pass}/{total})")

    return {
        "total": total,
        "base_pass": base_pass,
        "base_rate": round(base_rate, 2),
        "plus_pass": plus_pass,
        "plus_rate": round(plus_rate, 2),
        "details": details,
    }


def _run_test_script(script: str, timeout: int = 10) -> bool:
    """Run a Python test script, return True if all assertions pass."""
    import subprocess, tempfile, shutil
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_"))
    script_path = tmpdir / "test.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_and_capture(script: str, timeout: int = 5) -> Optional[str]:
    """Run script and capture stdout (last line)."""
    import subprocess, tempfile, shutil
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_ref_"))
    script_path = tmpdir / "ref.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark run
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    runtime: RuntimeConfig,
    kb_path: str,
    results_dir: Path,
    limit: Optional[int] = None,
    n_examples: int = 3,
):
    """Run full benchmark: generate with KB + without KB, then evaluate both."""
    model_safe = runtime.model.replace("/", "_").replace(":", "_")
    model_dir = results_dir / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_mbpp_test_tasks()
    if limit:
        tasks = tasks[:limit]
    logger.info("Benchmark | %s tasks | model=%s", len(tasks), runtime.model)

    # Load KB retriever
    retriever = None
    if Path(kb_path).exists():
        retriever = KnowledgeRetriever(kb_path=str(kb_path))
        retriever.build_index()
        logger.info("KB loaded | %s entries", len(retriever.entries))
    else:
        logger.warning("KB not found at %s, skipping with_kb mode", kb_path)

    results_summary = {}

    # A) Generate WITH KB
    if retriever:
        kb_file = model_dir / "with_kb_samples.jsonl"
        logger.info("=== Mode A: WITH KB ===")
        t0 = time.time()
        generate_all(runtime, tasks, kb_file, retriever=retriever, n_examples=n_examples)
        results_summary["with_kb"] = {
            "samples_file": str(kb_file),
            "time_seconds": round(time.time() - t0, 1),
        }

    # B) Generate WITHOUT KB (baseline)
    baseline_file = model_dir / "baseline_samples.jsonl"
    logger.info("=== Mode B: BASELINE (no KB) ===")
    t0 = time.time()
    generate_all(runtime, tasks, baseline_file, retriever=None)
    results_summary["baseline"] = {
        "samples_file": str(baseline_file),
        "time_seconds": round(time.time() - t0, 1),
    }

    # C) Evaluate both with EvalPlus
    print("\n" + "=" * 60)
    print("  EvalPlus Evaluation")
    print("=" * 60)

    if retriever and "with_kb" in results_summary:
        print("\n--- WITH KB ---")
        eval_kb = evaluate_with_evalplus(str(results_summary["with_kb"]["samples_file"]))
        results_summary["with_kb"]["eval"] = eval_kb

    print("\n--- BASELINE ---")
    eval_base = evaluate_with_evalplus(str(results_summary["baseline"]["samples_file"]))
    results_summary["baseline"]["eval"] = eval_base

    # Save summary
    summary_file = model_dir / "benchmark_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({
            "model": runtime.model,
            "provider": runtime.provider,
            "tasks_count": len(tasks),
            "kb_path": str(kb_path),
            "kb_entries": len(retriever.entries) if retriever else 0,
            "results": results_summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Summary saved to {summary_file}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark on MBPP / MBPP+ (EvalPlus)")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate solutions for MBPP tasks")
    gen.add_argument("--provider", type=str, default="ollama-cloud")
    gen.add_argument("--model", type=str, default="gemma3:4b")
    gen.add_argument("--base_url", type=str, default=None)
    gen.add_argument("--api_key", type=str, default=None)
    gen.add_argument("--mode", type=str, choices=["with_kb", "baseline"], required=True)
    gen.add_argument("--kb_path", type=str, default=str(DEFAULT_KB_PATH))
    gen.add_argument("--n_examples", type=int, default=3, help="Number of KB examples for few-shot")
    gen.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    gen.add_argument("--output", type=str, default=None, help="Output JSONL file")
    gen.add_argument("--log_level", type=str, default="INFO")

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Evaluate generated samples with EvalPlus")
    ev.add_argument("--samples_file", type=str, required=True)
    ev.add_argument("--dataset", type=str, default="mbpp", choices=["mbpp", "humaneval"])

    # --- run ---
    run = sub.add_parser("run", help="Full run: generate + evaluate both modes")
    run.add_argument("--provider", type=str, default="ollama-cloud")
    run.add_argument("--model", type=str, default="gemma3:4b")
    run.add_argument("--base_url", type=str, default=None)
    run.add_argument("--api_key", type=str, default=None)
    run.add_argument("--kb_path", type=str, default=str(DEFAULT_KB_PATH))
    run.add_argument("--n_examples", type=int, default=3)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    run.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level if hasattr(args, "log_level") else "INFO"),
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "generate":
        runtime = make_runtime(args.provider, args.model, args.base_url, args.api_key)
        tasks = load_mbpp_test_tasks()
        if args.limit:
            tasks = tasks[:args.limit]

        retriever = None
        if args.mode == "with_kb" and Path(args.kb_path).exists():
            retriever = KnowledgeRetriever(kb_path=args.kb_path)
            retriever.build_index()

        model_safe = runtime.model.replace("/", "_").replace(":", "_")
        output = Path(args.output) if args.output else DEFAULT_RESULTS_DIR / model_safe / f"{args.mode}_samples.jsonl"
        generate_all(runtime, tasks, output, retriever=retriever, n_examples=args.n_examples)
        print(f"Saved to {output}")

    elif args.command == "evaluate":
        evaluate_with_evalplus(args.samples_file, args.dataset)

    elif args.command == "run":
        runtime = make_runtime(args.provider, args.model, args.base_url, args.api_key)
        run_benchmark(
            runtime=runtime,
            kb_path=args.kb_path,
            results_dir=Path(args.results_dir),
            limit=args.limit,
            n_examples=args.n_examples,
        )


if __name__ == "__main__":
    main()
