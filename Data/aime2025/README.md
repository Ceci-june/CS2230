# AIME2025 Data & Scripts

Tài liệu chi tiết cho toàn bộ dữ liệu và script trong `Data/aime2025/`.
Mục tiêu là giúp bạn chọn đúng script theo môi trường (local macOS hay server GPU) và chạy được ngay.

## 1) Tổng quan nhanh

`Data/aime2025/` gồm 3 nhóm chính:

1. **Chuẩn bị dữ liệu**: `prepare_data.py`
2. **Đánh giá qua API** (chạy tốt trên macOS):
   - `eval_api.py` (không tool use)
   - `eval_api_tool.py` (có tool use mô phỏng Agent0)
3. **Pipeline Agent0 đầy đủ** (ưu tiên Linux + NVIDIA GPU):
   - `eval_aime2025.sh`
   - `train_with_aime2025_val.sh`
   - `run_agent0_train.sh`
   - `run_agent0_full.sh`
   - `run_train.sh`
   - `kaggle_train.py`

## 2) Ma trận tương thích môi trường

| Script | macOS CPU/MPS | Linux + NVIDIA GPU | Ghi chú |
|---|---|---|---|
| `prepare_data.py` | ✅ | ✅ | Tải và xuất parquet |
| `eval_api.py` | ✅ | ✅ | Đánh giá qua API |
| `eval_api_tool.py` | ✅ | ✅ | Có chạy code Python local |
| `eval_aime2025.sh` | ⚠️ Không khuyến nghị | ✅ | Cần stack Agent0 executor/eval service |
| `train_with_aime2025_val.sh` | ❌ | ✅ | Train ADPO trên VeRL |
| `run_agent0_train.sh` | ❌ | ✅ | Pipeline train cho 2x T4 |
| `run_agent0_full.sh` | ❌ | ✅ | Co-evolution nhiều vòng |
| `run_train.sh` | ⚠️ Tùy cấu hình | ✅ | Pipeline SFT/QLoRA đơn giản hơn |
| `kaggle_train.py` | ❌ | ✅ (Kaggle T4) | Notebook-oriented |

> Nếu bạn đang dùng MacBook, đường đi ổn định nhất là `prepare_data.py` + `eval_api.py` hoặc `eval_api_tool.py`.

## 3) Cài đặt tối thiểu

```bash
pip install datasets pandas pyarrow openai anthropic
```

Gói thường cần thêm cho một số script:

```bash
pip install mathruler stopit huggingface_hub
```

## 4) Chuẩn bị dữ liệu AIME2025

### Script: `prepare_data.py`

Chức năng:
- Tải `AIME2025-I` và `AIME2025-II` từ `opencompass/AIME2025`.
- Gộp thành bộ test 30 câu.
- Xuất 3 file:
  - `aime2025_test.parquet`
  - `aime2025_I_test.parquet`
  - `aime2025_II_test.parquet`

Chạy:

```bash
cd Data/aime2025
python prepare_data.py
```

## 5) Đánh giá qua API (không tool use)

### Script: `eval_api.py`

Hỗ trợ provider:
- `openai`
- `anthropic`
- `groq`
- `together`
- `custom` (OpenAI-compatible endpoint)

Các tham số quan trọng:
- `--provider`: chọn nhà cung cấp API.
- `--model`: tên model.
- `--data`: đường dẫn parquet (mặc định `aime2025_test.parquet`).
- `--samples`: số lần sample mỗi câu (dùng majority vote / pass@k).
- `--max_tokens`: trần số token output.
- `--output`: file JSON đầu ra.

Ví dụ:

```bash
cd Data/aime2025

export OPENAI_API_KEY="..."
python eval_api.py --provider openai --model gpt-4o

export GROQ_API_KEY="..."
python eval_api.py --provider groq --model qwen-qwq-32b --samples 3

python eval_api.py \
  --provider custom \
  --base_url http://localhost:8000/v1 \
  --api_key not-used \
  --model my-model
```

Đầu ra mặc định:
- `results_<provider>_<model>.json`

## 6) Đánh giá API + Tool Use (mô phỏng Agent0)

### Script: `eval_api_tool.py`

Điểm khác biệt so với `eval_api.py`:
- Cho model viết code trong khối ` ```python ... ``` `.
- Chạy code bằng subprocess Python local.
- Bơm output ngược vào hội thoại qua khối ` ```output ... ``` `.
- Hỗ trợ multi-turn reasoning (`--max_turns`, mặc định 4).

Các tham số quan trọng:
- `--provider`, `--model`, `--data`, `--output` tương tự `eval_api.py`.
- `--max_turns`: số vòng tool-call tối đa.
- `--samples`: số sample/câu để majority vote.

Ví dụ:

```bash
cd Data/aime2025

export GROQ_API_KEY="..."
python eval_api_tool.py --provider groq --model qwen/qwen3-32b

python eval_api_tool.py --provider groq --model qwen/qwen3-32b --samples 5 --max_turns 4

# Ollama local qua OpenAI-compatible API
python eval_api_tool.py \
  --provider custom \
  --base_url http://localhost:11434/v1 \
  --api_key not-used \
  --model qwen3:4b
```

Đầu ra mặc định:
- `results_tool_<provider>_<model>.json`

⚠️ Bảo mật:
- Script này thực thi code do model sinh ra trên máy local.
- Chỉ chạy trong môi trường bạn tin cậy.

## 7) Script Agent0 đầy đủ (GPU/server)

### `eval_aime2025.sh`

Mục đích:
- Bật tool server (`verl_tool.servers.serve`) và API service (`eval_service/app.py`) trong `executor_train`.
- Gọi endpoint nội bộ `http://localhost:5000/v1` để giải toàn bộ AIME2025.

Tham số:
- `$1`: `model_path` (mặc định `Qwen/Qwen3-4B-Base`)
- `$2`: `num_gpus` (mặc định `1`)

Ví dụ:

```bash
bash eval_aime2025.sh Qwen/Qwen3-4B-Base 1
```

Yêu cầu:
- Môi trường Agent0 đã cài đầy đủ.
- Linux + NVIDIA GPU (không phải luồng chạy tốt nhất cho macOS).

### `train_with_aime2025_val.sh`

Mục đích:
- Train Agent0 Executor bằng ADPO.
- Dùng file bạn truyền vào làm train set.
- Dùng `aime2025_I_test.parquet` + `aime2025_II_test.parquet` làm val set.

Tham số:
- `$1`: đường dẫn `train_data_parquet` (bắt buộc)
- `$2`: model name/path (optional)
- `$3`: số GPU (optional, mặc định 8)

Ví dụ:

```bash
bash train_with_aime2025_val.sh /path/to/train.parquet Qwen/Qwen3-4B-Base 8
```

### `run_agent0_train.sh`

Mục đích:
- Pipeline train/eval Agent0 đã chỉnh cho `2x T4 16GB`.

Mode:
- `setup`
- `data`
- `train`
- `eval`
- `all`

Ví dụ:

```bash
bash run_agent0_train.sh setup
bash run_agent0_train.sh all
```

### `run_agent0_full.sh`

Mục đích:
- Chạy full co-evolution Curriculum Agent ↔ Executor Agent theo nhiều iteration.

Mode phổ biến:
- `setup`, `test`, `all`
- `iter <n>`
- `curriculum <n>`, `generate <n>`, `executor <n>`, `eval <n>`

Ví dụ:

```bash
bash run_agent0_full.sh setup
bash run_agent0_full.sh iter 1
```

### `run_train.sh`

Mục đích:
- Pipeline đơn giản hơn theo hướng SFT/QLoRA.
- Có thể tạo training data từ Groq hoặc synthetic fallback.

Mode:
- `install`, `data`, `eval_base`, `train`, `eval_ft`, `all`

### `kaggle_train.py`

Mục đích:
- Phiên bản notebook-friendly cho Kaggle T4 x2.
- Distillation + QLoRA từ kết quả model lớn.

## 8) Cấu trúc dữ liệu và output

Input chính:
- `aime2025_test.parquet`
- `aime2025_I_test.parquet`
- `aime2025_II_test.parquet`

Kết quả mẫu:
- `results_custom_llama3.2:3b.json`
- `results_groq_qwen-qwq-32b.json`
- `results_groq_qwen_qwen3-32b.json`
- `results_tool_groq_qwen_qwen3-32b.json`

Ý nghĩa một bản ghi kết quả (rút gọn):
- `question`: đề bài.
- `ground_truth`: đáp án chuẩn.
- `final_answer` hoặc `extracted`: đáp án model sau khi parse `\boxed{}`.
- `correct`: đúng/sai.
- `samples`: danh sách từng lần sample (nếu `--samples > 1`).

## 9) Quick start đề xuất cho macOS

```bash
cd Data/aime2025
python prepare_data.py

export GROQ_API_KEY="..."
python eval_api.py --provider groq --model qwen-qwq-32b

python eval_api_tool.py --provider groq --model qwen/qwen3-32b
```

Nếu dùng Ollama local:

```bash
python eval_api_tool.py \
  --provider custom \
  --base_url http://localhost:11434/v1 \
  --api_key not-used \
  --model qwen3:4b
```

## 10) Lỗi thường gặp & cách xử lý

1. `Error: ... parquet not found`
   - Chạy lại `python prepare_data.py`.

2. `401 Unauthorized` hoặc `invalid api key`
   - Kiểm tra biến môi trường: `OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `ANTHROPIC_API_KEY`.

3. `Connection refused` với `--provider custom`
   - Kiểm tra service đang chạy và đúng `--base_url` (ví dụ `http://localhost:11434/v1`).

4. Script shell GPU fail trên macOS
   - Dùng `eval_api.py` hoặc `eval_api_tool.py` thay thế.

5. Tool-use chạy chậm
   - Giảm `--samples` hoặc `--max_turns`.

## 11) Gợi ý tái lập kết quả

- Luôn cố định `temperature=0.0` (đã đặt trong script).
- Lưu file output riêng bằng `--output` để dễ so sánh giữa model/provider.
- Khi so sánh công bằng, giữ nguyên `--samples`, `--max_turns`, `--max_tokens` giữa các lần chạy.
