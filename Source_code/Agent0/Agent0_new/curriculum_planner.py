#!/usr/bin/env python3
"""
Curriculum Planner — Phase 1 of Agent0 Synthetic Task Generation
================================================================

This module handles the planning phase: given only a high-level domain
(e.g. "coding"), the LLM autonomously:
  1) Proposes an initial set of subtopics with difficulty and task counts
  2) Reflects on the plan to identify gaps and missing topics
  3) Outputs a final, diverse curriculum plan as JSON

The plan is saved to a JSON file and can be consumed by
run_agent0_mbpp_curriculum.py (Phase 2: task generation).

Usage:
  # As standalone script
  python curriculum_planner.py --domain "coding" --total_tasks 30 --provider groq

  # As library
  from curriculum_planner import plan_curriculum
  plan = plan_curriculum(runtime, domain="coding", total_tasks=30, reflection_rounds=2)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Env loading
# ─────────────────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

def _load_env() -> None:
    if load_dotenv is None:
        return
    for candidate in [SCRIPT_DIR / ".env", REPO_ROOT / ".env", Path.cwd() / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
            break

_load_env()

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("curriculum_planner")

def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    quiet: bool = False,
) -> None:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING if quiet else logger.level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging initialized to %s", log_file)

# ─────────────────────────────────────────────────────────────────────────────
# Runtime config (lightweight copy — avoids circular import)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

@dataclass
class RuntimeConfig:
    provider: str
    model: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


def resolve_runtime(provider: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> RuntimeConfig:
    provider = provider.lower()
    if provider == "groq":
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        return RuntimeConfig(provider="groq", model=model, backend="Groq API", base_url=GROQ_BASE_URL, api_key=api_key)
    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return RuntimeConfig(provider="openai", model=model, backend="OpenAI API", base_url=base_url or OPENAI_BASE_URL, api_key=api_key)
    if provider == "ollama":
        return RuntimeConfig(provider="ollama", model=model, backend="Ollama local", base_url=base_url or OLLAMA_BASE_URL)
    if provider == "ollama-cloud":
        api_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not set")
        return RuntimeConfig(provider="ollama-cloud", model=model, backend="Ollama Cloud", base_url=base_url or OLLAMA_CLOUD_BASE_URL, api_key=api_key)
    if provider == "custom":
        return RuntimeConfig(provider="custom", model=model, backend="Custom API", base_url=base_url or OPENAI_BASE_URL, api_key=api_key or "not-used")
    raise ValueError(f"Unknown provider: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP + LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def http_post_json(url: str, payload: dict, timeout: int = 300, headers: Optional[dict] = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": "Agent0/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}: {err}") from e


def chat_completion(runtime: RuntimeConfig, messages: List[dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    # Ollama local and Ollama cloud both use /api/chat format
    if runtime.provider in ("ollama", "ollama-cloud"):
        payload = {"model": runtime.model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        headers = {}
        if runtime.api_key:
            headers["Authorization"] = f"Bearer {runtime.api_key}"
        url = f"{runtime.base_url.rstrip('/')}/api/chat"
        for _retry in range(3):
            try:
                body = http_post_json(url, payload, timeout=300, headers=headers)
                break
            except RuntimeError as e:
                if "429" in str(e) and _retry < 2:
                    logger.info("Rate limited, waiting 40s before retry...")
                    time.sleep(40)
                    continue
                raise
        else:
            raise RuntimeError(f"Rate limit exceeded after 3 retries")
        return (body.get("message", {}).get("content", "") or "").strip()

    # Groq / OpenAI / custom use OpenAI-compatible /chat/completions format
    assert runtime.base_url is not None
    payload = {"model": runtime.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {}
    if runtime.api_key and runtime.api_key != "not-used":
        headers["Authorization"] = f"Bearer {runtime.api_key}"
    url = f"{runtime.base_url.rstrip('/')}/chat/completions"
    for _retry in range(3):
        try:
            body = http_post_json(url, payload, timeout=300, headers=headers)
            break
        except RuntimeError as e:
            if "429" in str(e) and _retry < 2:
                logger.info("Rate limited, waiting 40s before retry...")
                time.sleep(40)
                continue
            raise
    else:
        raise RuntimeError(f"Rate limit exceeded after 3 retries")
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned by API")
    return (choices[0].get("message", {}).get("content", "") or "").strip()


def parse_json_from_llm(content: str) -> Any:
    """Parse JSON from LLM output, stripping think blocks and markdown."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        # Try to find JSON array or object in the text
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    continue
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Core planning logic
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a curriculum designer for coding education. "
        "Return strict JSON only, no markdown. No explanation."
    ),
}


def _parse_subtopics(content: str) -> List[dict]:
    """Parse LLM output into validated subtopic dicts."""
    parsed = parse_json_from_llm(content)
    if isinstance(parsed, dict):
        items = (parsed.get("subtopics") or parsed.get("new_subtopics")
                 or parsed.get("plan") or parsed.get("data") or [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    valid = []
    for st in items:
        if isinstance(st, dict) and st.get("name"):
            st.setdefault("task_count", 1)
            st.setdefault("difficulty", "medium")
            st.setdefault("description", st["name"])
            valid.append(st)
    return valid


def _adjust_task_counts(subtopics: List[dict], target: int) -> List[dict]:
    """Adjust task_count across subtopics so the total matches target."""
    total = sum(s["task_count"] for s in subtopics)
    if total != target:
        for s in subtopics:
            s["task_count"] = max(1, round(s["task_count"] * target / total))
        diff = target - sum(s["task_count"] for s in subtopics)
        subtopics[0]["task_count"] += diff
    return subtopics


def _initial_plan(runtime: RuntimeConfig, domain: str, count: int) -> List[dict]:
    """Step 1: LLM proposes an initial curriculum plan."""
    prompt = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Plan subtopics for a coding knowledge base",
                "domain": domain,
                "total_tasks": count,
                "instructions": [
                    f"Create a diverse curriculum plan with subtopics covering the domain '{domain}'.",
                    "For each subtopic, specify: name, description, difficulty (easy/medium/hard), and task_count.",
                    f"The sum of all task_count must equal {count}.",
                    "Cover a wide range: basics, data structures, algorithms, string manipulation, math, etc.",
                    "Return a JSON array of objects with keys: name, description, difficulty, task_count.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    logger.info("Step 1 — Initial plan | domain='%s' total_tasks=%s", domain, count)
    content = chat_completion(runtime, [SYSTEM_MSG, prompt])
    subtopics = _parse_subtopics(content)

    if not subtopics:
        raise ValueError("LLM failed to create initial plan — returned empty result")

    logger.info(
        "Step 1 done | %s subtopics: %s",
        len(subtopics),
        [s["name"] for s in subtopics],
    )
    return subtopics


def _reflect_on_plan(runtime: RuntimeConfig, domain: str, count: int, current: List[dict], round_num: int) -> List[dict]:
    """Step 2: LLM identifies MISSING topics and adds them to the plan."""
    current_names = [s["name"] for s in current]

    prompt = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Identify MISSING topics and add them to the curriculum",
                "domain": domain,
                "existing_subtopics": current_names,
                "instructions": [
                    f"The current plan has these subtopics: {current_names}.",
                    "Your job: identify important topics that are MISSING from this list.",
                    "Think about: recursion, dynamic programming, greedy algorithms, "
                    "bit manipulation, graph traversal (BFS/DFS), tree operations, "
                    "hash tables, linked lists, sorting algorithms, binary search, "
                    "backtracking, sliding window, two pointers, stack/queue problems, "
                    "heaps/priority queues, divide and conquer, memoization, etc.",
                    "Return ONLY the NEW subtopics as a JSON array (do NOT repeat existing ones).",
                    "Each new subtopic: name, description, difficulty (easy/medium/hard), task_count (1-3).",
                    "Add 3-6 new subtopics that are genuinely missing.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    logger.info(
        "Step 2 — Reflection round %s | reviewing %s subtopics: %s",
        round_num, len(current), current_names,
    )

    content = chat_completion(runtime, [SYSTEM_MSG, prompt])
    logger.debug("Reflection %s raw output:\n%s", round_num, content[:2000])
    new_subtopics = _parse_subtopics(content)
    logger.debug("Reflection %s parsed %s subtopics: %s", round_num, len(new_subtopics), [s["name"] for s in new_subtopics])

    # Filter out duplicates
    existing_lower = {n.lower() for n in current_names}
    added = [s for s in new_subtopics if s["name"].lower() not in existing_lower]

    if added:
        logger.info(
            "Step 2 — Reflection %s done | +%s new subtopics: %s",
            round_num, len(added), [s["name"] for s in added],
        )
        return current + added

    logger.warning("Step 2 — Reflection %s found no new subtopics to add", round_num)
    return current


def plan_curriculum(
    runtime: RuntimeConfig,
    domain: str = "coding",
    total_tasks: int = 30,
    reflection_rounds: int = 1,
) -> List[dict]:
    """
    Main entry point: plan a curriculum with reflection.

    Returns a list of subtopic dicts ready for task generation.
    """
    # Step 1: initial plan
    subtopics = _initial_plan(runtime, domain, total_tasks)

    # Step 2: reflection rounds
    for r in range(1, reflection_rounds + 1):
        try:
            subtopics = _reflect_on_plan(runtime, domain, total_tasks, subtopics, r)
        except Exception as e:
            logger.warning("Reflection round %s failed: %s", r, e)
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(40)

    # Adjust counts
    subtopics = _adjust_task_counts(subtopics, total_tasks)

    logger.info(
        "Planning complete | %s subtopics | plan=%s",
        len(subtopics),
        [(s["name"], s["task_count"], s["difficulty"]) for s in subtopics],
    )
    return subtopics

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum Planner — LLM plans coding subtopics with reflection")
    parser.add_argument("--domain", type=str, default="coding", help="High-level domain (default: coding)")
    parser.add_argument("--total_tasks", type=int, default=30, help="Total number of tasks to plan for")
    parser.add_argument("--reflection_rounds", type=int, default=1, help="Number of reflection rounds to improve the plan")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openai", "ollama", "ollama-cloud", "custom"])
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    setup_logger(log_file=args.log_file, log_level=args.log_level, quiet=args.quiet)

    runtime = resolve_runtime(args.provider, args.model, args.base_url, args.api_key)
    logger.info("Runtime: provider=%s model=%s", runtime.provider, runtime.model)

    plan = plan_curriculum(
        runtime=runtime,
        domain=args.domain,
        total_tasks=args.total_tasks,
        reflection_rounds=args.reflection_rounds,
    )

    output = {
        "domain": args.domain,
        "total_tasks": args.total_tasks,
        "reflection_rounds": args.reflection_rounds,
        "provider": runtime.provider,
        "model": runtime.model,
        "subtopics": plan,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Plan saved to {args.output}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"  Curriculum Plan — {args.domain}")
        print(f"{'='*60}")
        for i, s in enumerate(plan, 1):
            print(f"  {i:2d}. [{s['difficulty']:6s}] {s['name']} ({s['task_count']} tasks)")
            print(f"      {s['description']}")
        print(f"{'='*60}")
        print(f"  Total: {sum(s['task_count'] for s in plan)} tasks across {len(plan)} subtopics")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
