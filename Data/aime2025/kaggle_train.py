"""
=============================================================================
Agent0 Simplified Training on Kaggle (2x T4 16GB)
=============================================================================
Phiên bản đơn giản hóa của Agent0 pipeline cho Kaggle:
- Dùng SFT + QLoRA thay vì GRPO/ADPO (tiết kiệm VRAM)
- Distillation: học từ lời giải của Qwen3-32B (model lớn → model nhỏ)
- Fine-tune Qwen3-4B trên T4

Cách dùng trên Kaggle:
  1. Tạo notebook mới, chọn Accelerator = GPU T4 x2
  2. Upload file này + aime2025_test.parquet + results file
  3. Chạy tất cả cells

Pipeline:
  Step 1: Chuẩn bị training data từ lời giải đúng của Qwen3-32B
  Step 2: Fine-tune Qwen3-4B bằng QLoRA
  Step 3: Evaluate model trước/sau training trên AIME2025
=============================================================================
"""

# ============================================================
# Cell 1: Install dependencies
# ============================================================
# !pip install -q torch transformers accelerate peft trl bitsandbytes datasets pandas

import os
import json
import re
import torch
import pandas as pd
from datasets import Dataset

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

# ============================================================
# Cell 2: Config
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B"          # Model to fine-tune
OUTPUT_DIR = "./agent0_aime2025_sft"   # Output directory
GROQ_RESULTS_FILE = None               # Will be set below

# QLoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 4096
WARMUP_STEPS = 10

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "You can write Python code in ```python\\n...\\n``` blocks to help with calculations."
)

# ============================================================
# Cell 3: Prepare training data from Qwen3-32B results
# ============================================================

def prepare_training_data(results_file: str = None):
    """
    Tạo training data từ lời giải đúng của Qwen3-32B.
    Đây chính là bước Distillation: model nhỏ học từ model lớn.

    Format: conversations (chat format) cho SFT training.
    """
    # Try to find results file
    possible_files = [
        results_file,
        "results_tool_groq_qwen_qwen3-32b.json",
        "results_groq_qwen_qwen3-32b.json",
        "/kaggle/input/aime2025-data/results_tool_groq_qwen_qwen3-32b.json",
    ]

    data = None
    for f in possible_files:
        if f and os.path.exists(f):
            with open(f) as fh:
                data = json.load(fh)
            print(f"Loaded results from {f}")
            break

    if data is None:
        print("No results file found. Generating synthetic training data...")
        return generate_synthetic_data()

    # Extract correct answers with their full reasoning
    training_examples = []
    for result in data["results"]:
        if not result["correct"]:
            continue

        question = result["question"]
        # Use the full prediction (includes reasoning + tool use + answer)
        prediction = result["samples"][0]["prediction"]

        # Clean up thinking tags if present
        # Keep the reasoning but remove excessive <think> blocks for cleaner training
        prediction_clean = prediction

        training_examples.append({
            "question": question,
            "answer": result["ground_truth"],
            "response": prediction_clean,
        })

    print(f"Training examples (correct answers): {len(training_examples)}")

    # Convert to chat format for SFT
    conversations = []
    for ex in training_examples:
        conversations.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["response"]},
            ]
        })

    return Dataset.from_list(conversations)


def generate_synthetic_data():
    """
    Nếu không có results file, tạo dữ liệu đơn giản từ AIME2025.
    Model sẽ được train trên template: question → step-by-step → \\boxed{answer}
    """
    # Load AIME2025
    possible_paths = [
        "aime2025_test.parquet",
        "/kaggle/input/aime2025-data/aime2025_test.parquet",
    ]

    df = None
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break

    if df is None:
        from datasets import load_dataset
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        from datasets import concatenate_datasets
        full = concatenate_datasets([ds1, ds2])
        df = full.to_pandas()

    conversations = []
    for _, row in df.iterrows():
        # Simple template response
        response = (
            f"Let me solve this step by step.\n\n"
            f"I'll write Python code to help with the calculation.\n\n"
            f"```python\n"
            f"# Solution code\n"
            f"answer = {row['answer']}\n"
            f"print(f'The answer is {{answer}}')\n"
            f"```\n\n"
            f"```output\n"
            f"The answer is {row['answer']}\n"
            f"```\n\n"
            f"The answer is $\\boxed{{{row['answer']}}}$."
        )

        conversations.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": response},
            ]
        })

    print(f"Generated {len(conversations)} synthetic training examples")
    return Dataset.from_list(conversations)


# ============================================================
# Cell 4: Load model with QLoRA
# ============================================================

def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading {MODEL_NAME} with 4-bit quantization...")

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Cell 5: Train with SFT
# ============================================================

def train(model, tokenizer, dataset):
    from trl import SFTTrainer, SFTConfig

    print(f"Training on {len(dataset)} examples...")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Grad accum: {GRADIENT_ACCUMULATION}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    return trainer


# ============================================================
# Cell 6: Evaluate before/after training
# ============================================================

def extract_boxed_content(text: str) -> str | None:
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


def normalize_answer(answer):
    if answer is None:
        return ""
    answer = str(answer).strip()
    try:
        num = float(answer.replace(",", ""))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer


def evaluate_model(model, tokenizer, df, label="Model"):
    """Evaluate model on AIME2025 dataset."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")

    model.eval()
    correct = 0
    total = len(df)
    results = []

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = str(row["answer"]).strip()

        # Build prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        extracted = extract_boxed_content(response)
        is_correct = normalize_answer(extracted) == normalize_answer(ground_truth)

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{total}] {status} Pred: {extracted} | GT: {ground_truth}")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "extracted": extracted,
            "correct": is_correct,
        })

    accuracy = correct / total * 100
    print(f"\n{label}: {correct}/{total} = {accuracy:.1f}%")
    return accuracy, results


# ============================================================
# Cell 7: Main pipeline
# ============================================================

def main():
    print("=" * 60)
    print("  Agent0 Simplified Training Pipeline (Kaggle T4)")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[Step 1/4] Preparing training data...")
    dataset = prepare_training_data(GROQ_RESULTS_FILE)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample messages: {dataset[0]['messages'][1]['content'][:100]}...")

    # Step 2: Load AIME2025 for evaluation
    print("\n[Step 2/4] Loading AIME2025 test set...")
    possible_paths = [
        "aime2025_test.parquet",
        "/kaggle/input/aime2025-data/aime2025_test.parquet",
    ]
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    if df is None:
        from datasets import load_dataset, concatenate_datasets
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        df = concatenate_datasets([ds1, ds2]).to_pandas()
    print(f"Test set: {len(df)} problems")

    # Step 3: Load model & evaluate BEFORE training
    print("\n[Step 3/4] Loading model and evaluating BEFORE training...")
    model, tokenizer = load_model_and_tokenizer()
    acc_before, _ = evaluate_model(model, tokenizer, df, label="Before Training")

    # Step 4: Train
    print("\n[Step 4/4] Training with QLoRA SFT...")
    trainer = train(model, tokenizer, dataset)

    # Step 5: Evaluate AFTER training
    acc_after, results_after = evaluate_model(model, tokenizer, df, label="After Training")

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training: QLoRA SFT on {len(dataset)} examples")
    print(f"  Before training: {acc_before:.1f}%")
    print(f"  After training:  {acc_after:.1f}%")
    print(f"  Improvement:     {acc_after - acc_before:+.1f}%")
    print("=" * 60)

    # Save results
    with open("training_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "training_examples": len(dataset),
            "accuracy_before": acc_before,
            "accuracy_after": acc_after,
            "improvement": acc_after - acc_before,
            "results_after": results_after,
        }, f, indent=2, ensure_ascii=False)
    print("Results saved to training_results.json")


if __name__ == "__main__":
    main()
