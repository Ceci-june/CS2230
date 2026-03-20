"""
Đánh giá AIME2025 bằng API (OpenAI, Anthropic, Groq, Together AI, ...).
Chạy được trên MacBook - không cần GPU.

Cách dùng:
    # OpenAI
    export OPENAI_API_KEY="sk-..."
    python eval_api.py --provider openai --model gpt-4o

    # Anthropic (Claude)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python eval_api.py --provider anthropic --model claude-sonnet-4-20250514

    # Groq (miễn phí, chạy Qwen/Llama)
    export GROQ_API_KEY="gsk_..."
    python eval_api.py --provider groq --model qwen-qwq-32b

    # Together AI (chạy Qwen3, DeepSeek, ...)
    export TOGETHER_API_KEY="..."
    python eval_api.py --provider together --model Qwen/Qwen3-235B-A22B

    # Custom OpenAI-compatible endpoint
    python eval_api.py --provider custom --base_url http://localhost:8000/v1 --api_key not-used --model my-model
"""

import argparse
import json
import os
import re
import time

import pandas as pd

# ============================================================
# Grading logic (same as Agent0's math.py reward function)
# ============================================================

def extract_boxed_content(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    # Take the last \\boxed{} occurrence
    match = matches[-1]
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    # Remove leading zeros, trailing zeros after decimal
    try:
        num = float(answer.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    """Grade predicted answer against ground truth."""
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# ============================================================
# API Clients
# ============================================================

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "You can write Python code in ```python ... ``` blocks to help with calculations."
)


def call_openai(client, model: str, question: str, max_tokens: int = 16384) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_anthropic(client, model: str, question: str, max_tokens: int = 16384) -> str:
    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.content[0].text


def get_client_and_caller(provider: str, base_url: str = None, api_key: str = None):
    """Return (client, call_fn) based on provider."""
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        return client, call_anthropic

    # All others use OpenAI-compatible API
    from openai import OpenAI

    urls = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "custom": base_url,
    }

    keys = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
        "together": os.environ.get("TOGETHER_API_KEY"),
        "custom": api_key or "not-used",
    }

    client = OpenAI(
        api_key=api_key or keys.get(provider, ""),
        base_url=base_url or urls.get(provider),
    )
    return client, call_openai


# ============================================================
# Main evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate AIME2025 via API")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "groq", "together", "custom"])
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--data", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples per question (for pass@k / majority voting)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(script_dir, "aime2025_test.parquet")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} problems from {data_path}")

    # Setup API client
    client, call_fn = get_client_and_caller(args.provider, args.base_url, args.api_key)
    print(f"Provider: {args.provider} | Model: {args.model} | Samples: {args.samples}")
    print("=" * 60)

    results = []
    correct = 0
    total = len(df)

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = str(row["answer"]).strip()

        print(f"\n[{idx + 1}/{total}] {question[:80]}...")

        sample_answers = []
        for s in range(args.samples):
            try:
                prediction = call_fn(client, args.model, question, args.max_tokens)
                extracted = extract_boxed_content(prediction)
                is_correct = grade_answer(extracted, ground_truth)
                sample_answers.append({
                    "prediction": prediction,
                    "extracted": extracted,
                    "correct": is_correct,
                })
                status = "correct" if is_correct else "wrong"
                print(f"  Sample {s+1}: {extracted} ({status}) | GT: {ground_truth}")
            except Exception as e:
                print(f"  Sample {s+1}: ERROR - {e}")
                sample_answers.append({
                    "prediction": str(e),
                    "extracted": None,
                    "correct": False,
                })
            # Rate limit protection
            if args.samples > 1:
                time.sleep(0.5)

        # Majority voting if multiple samples
        if args.samples > 1:
            from collections import Counter
            valid_answers = [a["extracted"] for a in sample_answers if a["extracted"] is not None]
            if valid_answers:
                majority = Counter(valid_answers).most_common(1)[0][0]
                is_correct = grade_answer(majority, ground_truth)
            else:
                majority = None
                is_correct = False
            print(f"  Majority vote: {majority} | {'correct' if is_correct else 'wrong'}")
        else:
            is_correct = sample_answers[0]["correct"] if sample_answers else False
            majority = sample_answers[0].get("extracted") if sample_answers else None

        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": majority,
            "correct": is_correct,
            "samples": sample_answers,
        })

    # Summary
    accuracy = correct / total * 100
    print(f"\n{'=' * 60}")
    print(f"  AIME2025 Results ({args.provider}/{args.model})")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    if args.samples > 1:
        # Also compute pass@1
        pass1 = sum(any(s["correct"] for s in r["samples"]) for r in results)
        print(f"  Pass@{args.samples}: {pass1}/{total} ({pass1/total*100:.1f}%)")
    print(f"{'=' * 60}")

    # Save results
    output_path = args.output or os.path.join(
        script_dir, f"results_{args.provider}_{args.model.replace('/', '_')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": args.provider,
                "model": args.model,
                "samples": args.samples,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
