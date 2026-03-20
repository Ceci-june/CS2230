# Agent0: Self-Evolving Agents from Zero Data — Evaluation & Training on AIME2025

Đồ án môn CS2230: Reproduce và đánh giá framework Agent0 trên bộ dữ liệu AIME2025.

**Paper gốc**: [Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning](https://arxiv.org/abs/2511.16043)

**Dataset**: [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) — 30 bài toán từ kỳ thi AIME 2025 (American Invitational Mathematics Examination)

---

## Cấu trúc dự án

```
Do_an/
├── README.md                          # File này
├── .env                               # API keys (Groq)
├── .gitignore
├── Paper/                             # Paper gốc (PDF)
├── Source_code/
│   └── Agent0/                        # Source code Agent0 (clone từ GitHub)
└── Data/
    └── aime2025/                      # Dữ liệu + scripts
        ├── prepare_data.py            # Download AIME2025 → parquet
        ├── aime2025_test.parquet      # 30 bài AIME2025 (I + II)
        ├── aime2025_I_test.parquet    # 15 bài AIME2025-I
        ├── aime2025_II_test.parquet   # 15 bài AIME2025-II
        │
        │── # ====== EVALUATION (chạy trên MacBook / bất kỳ máy nào) ======
        ├── eval_api.py                # Eval qua API (không tool use)
        ├── eval_api_tool.py           # Eval qua API + code interpreter (giống paper)
        │
        │── # ====== TRAINING (cần GPU) ======
        ├── run_train.sh               # SFT + QLoRA (đơn giản, chắc chắn chạy)
        ├── run_agent0_train.sh        # Agent0 ADPO only (executor training)
        ├── run_agent0_full.sh         # Agent0 full pipeline (curriculum + executor)
        ├── Agent0_AIME2025_Kaggle.ipynb  # Notebook cho Kaggle
        ├── kaggle_train.py            # Script Python cho Kaggle
        │
        │── # ====== KẾT QUẢ ======
        ├── results_custom_llama3.2:3b.json          # Llama 3.2 3B: 0%
        ├── results_groq_qwen_qwen3-32b.json         # Qwen3-32B (no tool): 53.3%
        └── results_tool_groq_qwen_qwen3-32b.json    # Qwen3-32B (+ tool): 53.3%
```

---

## Yêu cầu

| Mục đích | Phần cứng | Phần mềm |
|---|---|---|
| Evaluation (API) | MacBook / bất kỳ | Python 3.10+, `openai`, `pandas` |
| Evaluation (local) | Ollama + bất kỳ | Ollama, model Qwen3/Llama |
| Training (SFT) | 1x T4 16GB | PyTorch, transformers, peft, trl |
| Training (Agent0 full) | 2x T4 16GB+ | Agent0 framework, vLLM, VeRL |

---

## Hướng dẫn chạy

### A. Evaluation qua API (không cần GPU)

Cách nhanh nhất để đánh giá model trên AIME2025.

#### 1. Chuẩn bị

```bash
cd Data/aime2025
pip install openai pandas
```

#### 2. Đặt API key

Tạo file `.env` ở thư mục gốc `Do_an/`:
```
GROQ_API_KEY="gsk_your_key_here"
```

Hoặc export trực tiếp:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

Lấy key miễn phí tại: https://console.groq.com/keys

#### 3. Chạy evaluation

```bash
# Không tool use (reasoning thuần)
python eval_api.py --provider groq --model qwen/qwen3-32b

# Có tool use / code interpreter (giống paper)
python eval_api_tool.py --provider groq --model qwen/qwen3-32b

# Majority voting (5 samples)
python eval_api.py --provider groq --model qwen/qwen3-32b --samples 5

# Dùng Ollama local
python eval_api.py --provider custom --base_url http://localhost:11434/v1 --api_key not-used --model qwen3:4b
```

#### 4. Kết quả

File JSON được lưu tự động, ví dụ: `results_groq_qwen_qwen3-32b.json`

Xem nhanh:
```bash
python3 -c "import json; d=json.load(open('results_groq_qwen_qwen3-32b.json')); print(f'Accuracy: {d[\"accuracy\"]:.1f}%')"
```

---

### B. Training SFT + QLoRA (1x T4, chắc chắn chạy)

Fine-tune model nhỏ (Qwen3-4B) bằng distillation từ model lớn (Qwen3-32B).

**Không dùng Agent0 framework** — dùng thư viện chuẩn (transformers, peft, trl).

#### Trên server (2x T4):

```bash
cd Data/aime2025
export GROQ_API_KEY="gsk_your_key_here"
bash run_train.sh all
```

Pipeline:
1. Cài thư viện
2. Download AIME2025
3. Gọi Qwen3-32B (Groq API) giải 30 bài → thu thập lời giải đúng
4. Eval Qwen3-4B trước training
5. Train QLoRA SFT
6. Eval sau training

#### Trên Kaggle (miễn phí):

1. Upload `aime2025_test.parquet` + `results_tool_groq_qwen_qwen3-32b.json` lên [Kaggle Datasets](https://www.kaggle.com/datasets/new)
2. Tạo notebook mới, chọn **GPU T4 x2**
3. Import file `Agent0_AIME2025_Kaggle.ipynb`
4. Chạy lần lượt từng cell

---

### C. Training Agent0 — Executor Only (2x T4)

Dùng Agent0 framework thật, chỉ train phần **Executor Agent** (ADPO + tool use).

```bash
# 1. Upload source code + data lên server
scp -r Source_code/Agent0/Agent0 user@server:~/Agent0/Agent0
scp -r Data/aime2025 user@server:~/aime2025

# 2. SSH vào server
ssh user@server
cd ~/aime2025

# 3. Chỉnh đường dẫn trong run_agent0_train.sh (dòng 21-23)
#    AGENT0_DIR="$HOME/Agent0/Agent0"
#    DATA_DIR="$HOME/aime2025"

# 4. Cài đặt + chạy
bash run_agent0_train.sh setup
bash run_agent0_train.sh all
```

---

### D. Training Agent0 — Full Pipeline (2x T4, giống paper)

**Full co-evolution**: Curriculum Agent ↔ Executor Agent, 3 iterations.

```bash
ssh user@server
cd ~/aime2025

# 1. Cài đặt
bash run_agent0_full.sh setup

# 2. Test GPU trước
bash run_agent0_full.sh test

# 3. Chạy full (3 iterations, ~6-8 tiếng)
bash run_agent0_full.sh all

# Hoặc chạy từng bước:
bash run_agent0_full.sh eval 0           # Eval base model
bash run_agent0_full.sh iter 1           # Iteration 1
bash run_agent0_full.sh iter 2           # Iteration 2
bash run_agent0_full.sh iter 3           # Iteration 3
```

Pipeline mỗi iteration:
```
Train Curriculum Agent (GRPO) → Sinh câu hỏi → Lọc (self-consistency)
→ Train Executor Agent (ADPO + tool use) → Eval AIME2025
```

**Lưu ý**: 2x T4 rất chật cho RL training. Nếu OOM, giảm config trong script hoặc dùng phương án B (SFT).

---

## Kết quả đã thu được

### Evaluation (không training)

| Model | Params | Tool Use | AIME2025 Accuracy |
|---|---|---|---|
| Llama 3.2 (Ollama local) | 3B | Không | 0/30 (0.0%) |
| Qwen3-32B (Groq API) | 32B | Không | 16/30 (53.3%) |
| Qwen3-32B (Groq API) | 32B | Có (code interpreter) | 16/30 (53.3%) |

### So sánh với paper

| Model | AIME2025 | Ghi chú |
|---|---|---|
| Qwen3-4B Base | 6.15% | Paper, no training |
| Qwen3-4B + Agent0 | **14.1%** | Paper, sau 3 iter RL |
| Qwen3-8B Base | 16.7% | Paper, no training |
| Qwen3-8B + Agent0 | **24.8%** | Paper, sau 3 iter RL |
| Qwen3-32B (mình eval) | **53.3%** | Zero-shot, no training |

---

## Giải thích pipeline Agent0

### Ý tưởng chính

Agent0 tạo 2 agent từ cùng 1 base model, cho chúng "thi đấu" để cùng tiến hóa:

```
Curriculum Agent (ra đề) ←→ Executor Agent (giải bài)
         ↓                           ↓
   GRPO Training              ADPO Training
   (học ra đề khó hơn)        (học giải bài + dùng tool)
```

### Các bước chi tiết

1. **Curriculum Agent Training (GRPO)**: Học cách sinh câu hỏi toán khó vừa phải — không quá dễ (executor giải hết) cũng không quá khó (executor không giải được)

2. **Question Generation**: Curriculum Agent sinh ~1000 câu hỏi mới

3. **Question Filtering**: Executor Agent thử giải, giữ lại câu hỏi có self-consistency score 0.3-0.8 (vùng "thử thách nhưng giải được")

4. **Executor Agent Training (ADPO)**: Học giải bài + dùng Python code interpreter qua multi-turn reasoning

5. **Co-evolution**: Lặp lại 3 vòng — curriculum sinh đề khó hơn, executor giải giỏi hơn

### File code quan trọng

| File | Chức năng |
|---|---|
| `Source_code/Agent0/Agent0/README.md` | Tổng quan |
| `curriculum_train/examples/reward_function/math.py` | Hàm chấm điểm (50 dòng) |
| `curriculum_train/examples/reward_function/curriculum_reward.py` | Reward cho curriculum agent |
| `curriculum_train/question_generate/question_generate.py` | Sinh câu hỏi |
| `curriculum_train/question_evaluate/evaluate.py` | Đánh giá câu hỏi |
| `executor_train/examples/train/math_tir/train_qwen3_4b_adpo.sh` | Script training executor |
| `executor_train/verl_tool/trainer/main_ppo.py` | Entry point RL training |

---

## Troubleshooting

### OOM (Out of Memory) khi training

```bash
# Giảm config trong run_agent0_full.sh:
batch_size=4          # giảm từ 8
n=2                   # giảm từ 4
max_response_length=1024  # giảm từ 2048
```

### vLLM không khởi động được

```bash
# Thử tăng gpu_memory_utilization
gpu_memory_utilization=0.45  # tăng từ 0.35
```

### Flash-attn không cài được trên T4

Bỏ qua — training vẫn chạy được, chỉ chậm hơn ~10%.

### Groq API rate limit

Free tier giới hạn 30 request/phút. Script đã có `time.sleep(1)` giữa các request. Nếu vẫn bị limit, tăng thời gian chờ hoặc dùng model nhỏ hơn (`llama-3.1-8b-instant`).

---

## Tài liệu tham khảo

- Paper: [Agent0 (arXiv:2511.16043)](https://arxiv.org/abs/2511.16043)
- Source code: [github.com/aiming-lab/Agent0](https://github.com/aiming-lab/Agent0)
- Dataset: [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025)
- VeRL framework: [github.com/volcengine/verl](https://github.com/volcengine/verl)
- VeRL-Tool: [github.com/TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)
