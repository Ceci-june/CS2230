# AIME2025 Data & Scripts

Tài liệu này mô tả các file dữ liệu/script trong `Data/aime2025/` để bạn chạy nhanh đúng mục tiêu (eval API, eval có tool-use, train Agent0).

## 1) Chuẩn bị môi trường

Tối thiểu cần Python 3.10+ và các gói:

```bash
pip install datasets pandas pyarrow openai anthropic
```

> Với script train Agent0 đầy đủ (`run_agent0_*.sh`, `train_with_aime2025_val.sh`), cần server Linux + NVIDIA GPU + môi trường Agent0/VeRL tương ứng.

## 2) Chuẩn bị dữ liệu AIME2025

Script: `prepare_data.py`

Chức năng:
- Tải `AIME2025-I` và `AIME2025-II` từ Hugging Face.
- Ghép thành bộ test 30 câu.
- Xuất ra các file parquet:
  - `aime2025_test.parquet`
  - `aime2025_I_test.parquet`
  - `aime2025_II_test.parquet`

Chạy:

```bash
cd Data/aime2025
python prepare_data.py
```

## 3) Đánh giá qua API (không tool use)

Script: `eval_api.py`

Đặc điểm:
- Hỗ trợ provider: `openai`, `anthropic`, `groq`, `together`, `custom`.
- Chấm điểm theo đáp án `\boxed{}` và normalize số học cơ bản.
- Hỗ trợ `--samples` để majority voting / pass@k.

Ví dụ:

```bash
cd Data/aime2025

# OpenAI
export OPENAI_API_KEY="..."
python eval_api.py --provider openai --model gpt-4o

# Groq
export GROQ_API_KEY="..."
python eval_api.py --provider groq --model qwen-qwq-32b

# Custom OpenAI-compatible endpoint (ví dụ local server)
python eval_api.py --provider custom --base_url http://localhost:8000/v1 --api_key not-used --model my-model
```

Output mặc định:
- `results_<provider>_<model>.json`

## 4) Đánh giá API + Tool Use (mô phỏng Agent0)

Script: `eval_api_tool.py`

Đặc điểm:
- Multi-turn reasoning (mặc định tối đa `--max_turns 4`).
- Tự trích code trong khối ```python ...``` và chạy local sandbox (`python3` subprocess).
- Bơm kết quả thực thi vào hội thoại qua khối ```output ...```.
- Chấm đáp án cuối cùng qua `\boxed{}`.

Ví dụ:

```bash
cd Data/aime2025

# Groq + tool use
export GROQ_API_KEY="..."
python eval_api_tool.py --provider groq --model qwen/qwen3-32b

# Majority voting
python eval_api_tool.py --provider groq --model qwen/qwen3-32b --samples 5

# Ollama local (OpenAI-compatible endpoint)
python eval_api_tool.py --provider custom --base_url http://localhost:11434/v1 --api_key not-used --model qwen3:4b
```

Output mặc định:
- `results_tool_<provider>_<model>.json`

## 5) Các script shell chính

### `eval_aime2025.sh`
- Đánh giá Agent0 executor service trên `aime2025_test.parquet`.
- Tự bật tool server + API service rồi gọi qua OpenAI-compatible endpoint nội bộ (`http://localhost:5000/v1`).
- Phù hợp môi trường đã setup Agent0 đầy đủ.

### `train_with_aime2025_val.sh`
- Huấn luyện Agent0 Executor (ADPO) với:
  - train file bạn truyền vào,
  - validation set là `aime2025_I_test.parquet` + `aime2025_II_test.parquet`.
- Dùng `verl_tool.trainer.main_ppo` và tool server python code.

### `run_agent0_train.sh`
- Pipeline train/eval Agent0 đã giảm cấu hình cho `2x T4 16GB`.
- Có các mode `setup`, `data`, `train`, `eval`, `all`.

### `run_agent0_full.sh`
- Pipeline co-evolution đầy đủ Curriculum Agent ↔ Executor Agent (nhiều iteration).
- Bao gồm train curriculum, generate question, evaluate/filter, train executor, eval AIME2025.

### `run_train.sh`
- Pipeline đơn giản hơn (SFT/QLoRA oriented), phù hợp baseline hoặc thử nghiệm nhanh.
- Có thể sinh training data từ Groq (`qwen/qwen3-32b`) hoặc synthetic fallback.

### `kaggle_train.py`
- Bản đơn giản hóa cho môi trường Kaggle (2x T4), theo hướng distillation + QLoRA.

## 6) Dữ liệu và kết quả trong thư mục

- Input parquet:
  - `aime2025_test.parquet`
  - `aime2025_I_test.parquet`
  - `aime2025_II_test.parquet`
- File kết quả mẫu:
  - `results_custom_llama3.2:3b.json`
  - `results_groq_qwen-qwq-32b.json`
  - `results_groq_qwen_qwen3-32b.json`
  - `results_tool_groq_qwen_qwen3-32b.json`

## 7) Quick start đề xuất (local macOS)

```bash
cd Data/aime2025
python prepare_data.py

# Đánh giá nhanh qua API thường
export GROQ_API_KEY="..."
python eval_api.py --provider groq --model qwen-qwq-32b

# Đánh giá có tool use
python eval_api_tool.py --provider groq --model qwen/qwen3-32b
```

## 8) Lưu ý vận hành

- `eval_api_tool.py` chạy code Python sinh bởi model trên máy local; chỉ dùng khi bạn chấp nhận rủi ro thực thi code.
- Nếu endpoint custom không tương thích hoàn toàn OpenAI API, có thể cần điều chỉnh `base_url`, tên model hoặc thông số token.
- Một số script train giả định có `conda`, `nvidia-smi`, và thư mục Agent0 tại `Source_code/Agent0/Agent0`.
