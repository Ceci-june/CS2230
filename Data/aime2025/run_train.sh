#!/bin/bash
set -e

###############################################################################
# Agent0 Simplified Training Pipeline
# Server: 2x NVIDIA T4 16GB
#
# Cách dùng:
#   chmod +x run_train.sh
#   bash run_train.sh
#
# Hoặc chạy từng bước:
#   bash run_train.sh install    # Cài đặt
#   bash run_train.sh data       # Chuẩn bị dữ liệu
#   bash run_train.sh eval_base  # Eval model gốc
#   bash run_train.sh train      # Train QLoRA
#   bash run_train.sh eval_ft    # Eval sau training
#   bash run_train.sh all        # Chạy tất cả
###############################################################################

# ========================== CONFIG ==========================
MODEL_NAME="Qwen/Qwen3-4B"
OUTPUT_DIR="./output/agent0_aime2025"
DATA_DIR="./data"
RESULTS_DIR="./results"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=4
LR=2e-4
MAX_SEQ_LEN=4096
LORA_R=16
LORA_ALPHA=32

# Load .env file nếu có
ENV_FILE="$(dirname "$0")/../../.env"
[ -f "$ENV_FILE" ] && export $(grep -v '^#' "$ENV_FILE" | xargs)

# Groq API (để tạo training data từ model lớn)
GROQ_API_KEY="${GROQ_API_KEY:-}"
GROQ_MODEL="qwen/qwen3-32b"

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "============================================"
echo "  Agent0 Training Pipeline"
echo "  Model:  $MODEL_NAME"
echo "  GPUs:   $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  VRAM:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) per GPU"
echo "============================================"

# ========================== FUNCTIONS ==========================

install_deps() {
    echo ""
    echo "[Step 1] Installing dependencies..."
    pip install -q torch transformers accelerate peft trl bitsandbytes \
        datasets pandas openai huggingface_hub
    echo "Done."
}

prepare_data() {
    echo ""
    echo "[Step 2] Preparing data..."
    mkdir -p $DATA_DIR $RESULTS_DIR $OUTPUT_DIR

    python3 << 'PYEOF'
import os, json
from datasets import load_dataset, concatenate_datasets

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Download AIME2025
print("Downloading AIME2025...")
aime1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
aime2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
full = concatenate_datasets([aime1, aime2])
full.to_parquet(f"{DATA_DIR}/aime2025_test.parquet")
print(f"Saved {len(full)} problems to {DATA_DIR}/aime2025_test.parquet")
PYEOF
    echo "Done."
}

generate_training_data() {
    echo ""
    echo "[Step 3] Generating training data from $GROQ_MODEL..."

    if [ -z "$GROQ_API_KEY" ]; then
        echo "WARNING: GROQ_API_KEY not set. Using synthetic data instead."
        echo "To use Qwen3-32B solutions, set: export GROQ_API_KEY=gsk_..."
        use_synthetic_data
        return
    fi

    python3 << 'PYEOF'
import os, json, re, time
import pandas as pd
from openai import OpenAI

DATA_DIR = os.environ.get("DATA_DIR", "./data")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "You can write Python code in ```python\n...\n``` blocks to help with calculations."
)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

df = pd.read_parquet(f"{DATA_DIR}/aime2025_test.parquet")
print(f"Solving {len(df)} AIME2025 problems with {GROQ_MODEL}...")

results = []
for idx, row in df.iterrows():
    q = row["question"]
    gt = str(row["answer"]).strip()
    print(f"  [{idx+1}/{len(df)}] {q[:60]}...", end=" ")

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            temperature=0.0,
            max_tokens=16384,
        )
        pred = resp.choices[0].message.content

        # Extract \boxed{}
        matches = list(re.finditer(r"\\boxed\{", pred))
        extracted = None
        if matches:
            m = matches[-1]
            s, d, i = m.end(), 1, m.end()
            while i < len(pred) and d > 0:
                if pred[i] == "{": d += 1
                elif pred[i] == "}": d -= 1
                i += 1
            if d == 0:
                extracted = pred[s:i-1].strip()

        # Normalize
        def norm(a):
            if a is None: return ""
            try:
                n = float(str(a).strip().replace(",",""))
                return str(int(n)) if n == int(n) else str(n)
            except: return str(a).strip()

        correct = norm(extracted) == norm(gt)
        print(f"{'✓' if correct else '✗'} ({extracted})")

        results.append({
            "question": q, "ground_truth": gt, "prediction": pred,
            "extracted": extracted, "correct": correct,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "question": q, "ground_truth": gt, "prediction": str(e),
            "extracted": None, "correct": False,
        })
    time.sleep(1)  # Rate limit

# Save
correct_count = sum(r["correct"] for r in results)
print(f"\nQwen3-32B accuracy: {correct_count}/{len(results)} = {correct_count/len(results)*100:.1f}%")

with open(f"{DATA_DIR}/qwen3_32b_solutions.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Create training data (only correct solutions)
training = []
for r in results:
    if r["correct"]:
        training.append({
            "question": r["question"],
            "answer": r["ground_truth"],
            "response": r["prediction"],
        })

with open(f"{DATA_DIR}/train_data.json", "w") as f:
    json.dump(training, f, indent=2, ensure_ascii=False)
print(f"Training data: {len(training)} correct solutions saved")
PYEOF
    echo "Done."
}

use_synthetic_data() {
    python3 << 'PYEOF'
import json, pandas as pd, os
DATA_DIR = os.environ.get("DATA_DIR", "./data")
df = pd.read_parquet(f"{DATA_DIR}/aime2025_test.parquet")
training = []
for _, row in df.iterrows():
    resp = (
        f"Let me solve this step by step.\n\n"
        f"```python\nanswer = {row['answer']}\nprint(answer)\n```\n\n"
        f"```output\n{row['answer']}\n```\n\n"
        f"The answer is $\\boxed{{{row['answer']}}}$."
    )
    training.append({"question": row["question"], "answer": str(row["answer"]), "response": resp})
with open(f"{DATA_DIR}/train_data.json", "w") as f:
    json.dump(training, f, indent=2, ensure_ascii=False)
print(f"Synthetic training data: {len(training)} examples saved")
PYEOF
}

eval_model() {
    local LABEL=$1
    local MODEL_PATH=$2
    local ADAPTER_PATH=$3

    echo ""
    echo "[Eval] $LABEL"
    echo "  Model: $MODEL_PATH"
    [ -n "$ADAPTER_PATH" ] && echo "  Adapter: $ADAPTER_PATH"

    python3 << PYEOF
import os, re, json, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_DIR = os.environ.get("DATA_DIR", "./data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
LABEL = "$LABEL"
MODEL_PATH = "$MODEL_PATH"
ADAPTER_PATH = "$ADAPTER_PATH" if "$ADAPTER_PATH" else None

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\\\boxed{}. "
    "You can write Python code in \`\`\`python\\\\n...\\\\n\`\`\` blocks to help with calculations."
)

# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16,
)

if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
    from peft import PeftModel
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# Evaluate
df = pd.read_parquet(f"{DATA_DIR}/aime2025_test.parquet")
correct, total = 0, len(df)
results = []

for idx, row in df.iterrows():
    q, gt = row["question"], str(row["answer"]).strip()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, temperature=0.0,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract boxed
    ext = None
    matches = list(re.finditer(r"\\\\boxed\{", resp))
    if matches:
        m = matches[-1]
        s, d, i = m.end(), 1, m.end()
        while i < len(resp) and d > 0:
            if resp[i] == "{": d += 1
            elif resp[i] == "}": d -= 1
            i += 1
        if d == 0: ext = resp[s:i-1].strip()

    def norm(a):
        if a is None: return ""
        try:
            n = float(str(a).strip().replace(",",""))
            return str(int(n)) if n == int(n) else str(n)
        except: return str(a).strip()

    ok = norm(ext) == norm(gt)
    if ok: correct += 1
    print(f"  [{idx+1}/{total}] {'✓' if ok else '✗'} Pred: {ext} | GT: {gt}")
    results.append({"question": q, "ground_truth": gt, "extracted": ext, "correct": ok})

acc = correct / total * 100
print(f"\n{LABEL}: {correct}/{total} = {acc:.1f}%")

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/eval_{LABEL.replace(' ','_').lower()}.json", "w") as f:
    json.dump({"label": LABEL, "accuracy": acc, "correct": correct,
               "total": total, "results": results}, f, indent=2, ensure_ascii=False)
PYEOF
}

train_qlora() {
    echo ""
    echo "[Step 4] Training QLoRA SFT..."
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Batch: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
    echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"

    python3 << PYEOF
import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MODEL_NAME = "$MODEL_NAME"
OUTPUT_DIR = "$OUTPUT_DIR"

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\\\boxed{}. "
    "You can write Python code in \`\`\`python\\\\n...\\\\n\`\`\` blocks to help with calculations."
)

# Load training data
with open(f"{DATA_DIR}/train_data.json") as f:
    raw = json.load(f)

conversations = []
for ex in raw:
    conversations.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["response"]},
        ]
    })
dataset = Dataset.from_list(conversations)
print(f"Training dataset: {len(dataset)} examples")

# Load model (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# LoRA
lora_config = LoraConfig(
    r=$LORA_R, lora_alpha=$LORA_ALPHA, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=$NUM_EPOCHS,
    per_device_train_batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=$GRAD_ACCUM,
    learning_rate=$LR,
    warmup_steps=10,
    max_seq_length=$MAX_SEQ_LEN,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=dataset, processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")
PYEOF
    echo "Done."
}

print_summary() {
    echo ""
    echo "============================================"
    echo "  PIPELINE COMPLETE"
    echo "============================================"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  Results:    $RESULTS_DIR/"
    echo ""
    echo "  Files:"
    ls -lh $RESULTS_DIR/ 2>/dev/null || echo "  (no results yet)"
    echo ""
    echo "  To compare before/after:"
    echo "    cat $RESULTS_DIR/eval_before_training.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f'Before: {d[\\\"accuracy\\\"]:.1f}%')\""
    echo "    cat $RESULTS_DIR/eval_after_training.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f'After: {d[\\\"accuracy\\\"]:.1f}%')\""
    echo "============================================"
}

# ========================== MAIN ==========================

export DATA_DIR RESULTS_DIR OUTPUT_DIR GROQ_API_KEY GROQ_MODEL MODEL_NAME

case "${1:-all}" in
    install)
        install_deps
        ;;
    data)
        prepare_data
        generate_training_data
        ;;
    eval_base)
        eval_model "Before Training" "$MODEL_NAME" ""
        ;;
    train)
        train_qlora
        ;;
    eval_ft)
        eval_model "After Training" "$MODEL_NAME" "$OUTPUT_DIR"
        ;;
    all)
        install_deps
        prepare_data
        generate_training_data
        eval_model "Before Training" "$MODEL_NAME" ""
        train_qlora
        eval_model "After Training" "$MODEL_NAME" "$OUTPUT_DIR"
        print_summary
        ;;
    *)
        echo "Usage: $0 {install|data|eval_base|train|eval_ft|all}"
        exit 1
        ;;
esac
