"""
Đánh giá AIME2025 bằng API + Tool Use (Code Interpreter).
Mô phỏng đúng cách Agent0 chạy trong paper:
  - Model viết code Python trong ```python...```
  - Code được chạy trong sandbox local
  - Kết quả trả về qua ```output...```
  - Model tiếp tục suy luận (multi-turn, tối đa max_turns lượt)
  - Cuối cùng trích đáp án từ \\boxed{}

Cách dùng:
    export GROQ_API_KEY="gsk_..."
    python eval_api_tool.py --provider groq --model qwen/qwen3-32b

    # Với majority voting
    python eval_api_tool.py --provider groq --model qwen/qwen3-32b --samples 5

    # Ollama local
    python eval_api_tool.py --provider custom --base_url http://localhost:11434/v1 --api_key not-used --model qwen3:4b
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
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
    if answer is None:
        return ""
    answer = answer.strip()
    try:
        num = float(answer.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# ============================================================
# Sandbox: Local Python code execution
# ============================================================

def execute_python_code(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a subprocess and return stdout/stderr.
    This mimics Agent0's SandboxFusion code interpreter.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            # Include error but truncate if too long
            stderr = result.stderr[:500]
            output = output + ("\n" + stderr if output else stderr)
        if not output.strip():
            output = "(no output)"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(tmp_path)


def extract_python_code(text: str) -> str | None:
    """
    Extract Python code from ```python...``` blocks.
    Same pattern Agent0 uses: code between ```python and ```.
    """
    # Match the last ```python...``` block
    pattern = r"```python\s*\n(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


# ============================================================
# Multi-turn API call with tool use
# ============================================================

SYSTEM_PROMPT_TOOL = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n"
    "You have access to a Python code interpreter. To use it, write code in a ```python\\n...\\n``` block. "
    "The code will be executed and you will receive the output. "
    "You can use this to perform calculations, verify your reasoning, or explore the problem.\n"
    "After receiving the code output, continue your reasoning and provide the final answer in \\boxed{}."
)


def call_with_tools(client, model: str, question: str, max_tokens: int = 16384,
                    max_turns: int = 4, provider: str = "openai") -> dict:
    """
    Multi-turn conversation with code execution, mimicking Agent0's pipeline:
      1. Model reasons and optionally writes ```python...``` code
      2. Code is executed locally
      3. Output is fed back as ```output\n...\n```
      4. Repeat up to max_turns times
      5. Return full conversation and final response
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOL},
        {"role": "user", "content": question},
    ]

    full_response = ""
    tool_calls = 0

    for turn in range(max_turns + 1):  # +1 for final answer turn
        try:
            if provider == "anthropic":
                from anthropic import Anthropic
                response = client.messages.create(
                    model=model,
                    system=SYSTEM_PROMPT_TOOL,
                    messages=[m for m in messages if m["role"] != "system"],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                assistant_msg = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                assistant_msg = response.choices[0].message.content
        except Exception as e:
            return {
                "full_response": full_response + f"\n[ERROR: {e}]",
                "tool_calls": tool_calls,
                "turns": turn,
            }

        full_response += assistant_msg

        # Check for Python code to execute
        code = extract_python_code(assistant_msg)

        if code and turn < max_turns:
            # Execute code in sandbox
            tool_calls += 1
            output = execute_python_code(code)

            # Truncate output if too long
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"

            # Format output like Agent0: ```output\n...\n```
            tool_response = f"```output\n{output}\n```"
            full_response += "\n" + tool_response + "\n"

            # Add to conversation
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": tool_response + "\nContinue your reasoning based on the output above. Put your final answer in \\boxed{}."})

            print(f"    [Turn {turn+1}] Executed code → {output[:80]}...")
        else:
            # No code or reached max turns → done
            break

    return {
        "full_response": full_response,
        "tool_calls": tool_calls,
        "turns": turn + 1,
    }


# ============================================================
# API client setup
# ============================================================

def get_client(provider: str, base_url: str = None, api_key: str = None):
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

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
    return OpenAI(
        api_key=api_key or keys.get(provider, ""),
        base_url=base_url or urls.get(provider),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate AIME2025 via API + Tool Use")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "groq", "together", "custom"])
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Max tool-calling turns (default: 4, same as paper)")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(script_dir, "aime2025_test.parquet")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} problems from {data_path}")

    # Setup
    client = get_client(args.provider, args.base_url, args.api_key)
    print(f"Provider: {args.provider} | Model: {args.model}")
    print(f"Samples: {args.samples} | Max turns: {args.max_turns} | Tool use: ENABLED")
    print("=" * 60)

    results = []
    correct = 0
    total = len(df)
    total_tool_calls = 0

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = str(row["answer"]).strip()

        print(f"\n[{idx + 1}/{total}] {question[:80]}...")

        sample_answers = []
        for s in range(args.samples):
            try:
                result = call_with_tools(
                    client, args.model, question,
                    max_tokens=args.max_tokens,
                    max_turns=args.max_turns,
                    provider=args.provider,
                )

                prediction = result["full_response"]
                extracted = extract_boxed_content(prediction)
                is_correct = grade_answer(extracted, ground_truth)
                total_tool_calls += result["tool_calls"]

                sample_answers.append({
                    "prediction": prediction,
                    "extracted": extracted,
                    "correct": is_correct,
                    "tool_calls": result["tool_calls"],
                    "turns": result["turns"],
                })
                status = "correct" if is_correct else "wrong"
                print(f"  Sample {s+1}: {extracted} ({status}) | GT: {ground_truth} | tools: {result['tool_calls']}")

            except Exception as e:
                print(f"  Sample {s+1}: ERROR - {e}")
                sample_answers.append({
                    "prediction": str(e),
                    "extracted": None,
                    "correct": False,
                    "tool_calls": 0,
                    "turns": 0,
                })

            if args.samples > 1:
                time.sleep(0.5)

        # Majority voting
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
    avg_tools = total_tool_calls / (total * args.samples) if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  AIME2025 Results ({args.provider}/{args.model})")
    print(f"  Mode: Tool-Integrated Reasoning (max {args.max_turns} turns)")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Avg tool calls/question: {avg_tools:.1f}")
    if args.samples > 1:
        pass1 = sum(any(s["correct"] for s in r["samples"]) for r in results)
        print(f"  Pass@{args.samples}: {pass1}/{total} ({pass1/total*100:.1f}%)")
    print(f"{'=' * 60}")

    # Save
    output_path = args.output or os.path.join(
        script_dir, f"results_tool_{args.provider}_{args.model.replace('/', '_')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": args.provider,
                "model": args.model,
                "mode": "tool_integrated_reasoning",
                "max_turns": args.max_turns,
                "samples": args.samples,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "avg_tool_calls": avg_tools,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
