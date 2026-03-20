#!/bin/bash
set -x

###############################################################################
# Script đánh giá Agent0 trên bộ dữ liệu AIME2025
#
# Cách sử dụng:
#   bash eval_aime2025.sh <model_path> [num_gpus]
#
# Ví dụ:
#   bash eval_aime2025.sh Qwen/Qwen3-4B-Base 1
#   bash eval_aime2025.sh /path/to/your/trained/model 2
###############################################################################

# === Tham số ===
model_path=${1:-"Qwen/Qwen3-4B-Base"}
num_gpus=${2:-1}

# === Đường dẫn ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT0_DIR="$(cd "$SCRIPT_DIR/../../Source_code/Agent0/Agent0" && pwd)"
DATA_DIR="$SCRIPT_DIR"

# File test AIME2025 (30 bài = AIME2025-I + AIME2025-II)
test_data="$DATA_DIR/aime2025_test.parquet"

# Kiểm tra file tồn tại
if [ ! -f "$test_data" ]; then
    echo "Error: $test_data not found. Run prepare_data.py first."
    echo "  cd $DATA_DIR && python prepare_data.py"
    exit 1
fi

echo "============================================"
echo "  Agent0 Evaluation on AIME2025"
echo "  Model: $model_path"
echo "  Data:  $test_data"
echo "  GPUs:  $num_gpus"
echo "============================================"

# === Chạy Evaluation Service ===
cd "$AGENT0_DIR/executor_train"

# 1. Start tool server (sandbox cho code execution)
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url="http://$host:$port/get_observation"
python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "python_code" \
    --workers_per_tool 8 \
    --done_if_invalid True \
    --slient True &
server_pid=$!
echo "Tool server (pid=$server_pid) started at $tool_server_url"
sleep 3

# 2. Start API service
api_host="0.0.0.0"
api_port=5000
max_turns=4
min_turns=0
action_stop_tokens='```output'
tensor_parallel_size=$num_gpus
num_models=1
enable_mtrl=False

# Temp file for action stop tokens
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file

python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool_server_url $tool_server_url \
    --model $model_path \
    --max_turns $max_turns \
    --min_turns $min_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor_parallel_size $tensor_parallel_size \
    --num_models $num_models \
    --enable_mtrl $enable_mtrl &
api_pid=$!
echo "API service (pid=$api_pid) started at $api_host:$api_port"
echo "Waiting for model to load..."
sleep 30

# 3. Chạy evaluation trên AIME2025
echo ""
echo "============================================"
echo "  Running evaluation..."
echo "============================================"

python - <<'EVAL_SCRIPT'
import json
import pandas as pd
from openai import OpenAI
from mathruler.grader import extract_boxed_content, grade_answer

# Load AIME2025 data
import sys, os
data_dir = os.environ.get("DATA_DIR", ".")
df = pd.read_parquet(os.path.join(data_dir, "aime2025_test.parquet"))

client = OpenAI(
    api_key="not-used",
    base_url="http://localhost:5000/v1"
)

results = []
correct = 0
total = len(df)

for idx, row in df.iterrows():
    question = row["question"]
    ground_truth = str(row["answer"])

    print(f"\n[{idx+1}/{total}] Solving: {question[:80]}...")

    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        prediction = response.choices[0].message.content
        answer = extract_boxed_content(prediction)

        try:
            is_correct = grade_answer(answer, ground_truth)
        except:
            is_correct = False

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "extracted_answer": answer,
            "correct": is_correct
        })
        print(f"  Answer: {answer} | Ground Truth: {ground_truth} | {'✓' if is_correct else '✗'}")

    except Exception as e:
        print(f"  Error: {e}")
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": str(e),
            "extracted_answer": None,
            "correct": False
        })

# Summary
accuracy = correct / total * 100
print(f"\n{'='*50}")
print(f"  AIME2025 Results")
print(f"  Correct: {correct}/{total}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"{'='*50}")

# Save results
output_path = os.path.join(data_dir, "aime2025_results.json")
with open(output_path, "w") as f:
    json.dump({"accuracy": accuracy, "correct": correct, "total": total, "results": results}, f, indent=2, ensure_ascii=False)
print(f"Results saved to {output_path}")
EVAL_SCRIPT

# 4. Cleanup
echo "Cleaning up..."
kill -9 $api_pid 2>/dev/null
kill -9 $server_pid 2>/dev/null
rm -f $action_stop_tokens_file

echo "Done!"
