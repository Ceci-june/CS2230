# Data

Thư mục này chứa dữ liệu và script phục vụ huấn luyện/đánh giá cho đồ án Agent0.

## Cấu trúc

```text
Data/
  aime2025/
    prepare_data.py
    eval_api.py
    eval_api_tool.py
    eval_aime2025.sh
    train_with_aime2025_val.sh
    run_agent0_train.sh
    run_agent0_full.sh
    run_train.sh
    kaggle_train.py
    *.parquet
    results_*.json
```

## Mục đích

- Tập trung vào benchmark **AIME2025**.
- Chuẩn bị dữ liệu từ Hugging Face (`opencompass/AIME2025`) sang định dạng `.parquet`.
- Đánh giá mô hình theo 2 chế độ:
  - API thông thường (single-turn/majority voting).
  - API + tool use (mô phỏng code interpreter nhiều lượt).
- Cung cấp script train/eval nhanh và pipeline mở rộng cho Agent0.

## Thành phần chính

- `aime2025/README.md`: tài liệu chi tiết cho toàn bộ script trong bộ AIME2025.

## Luồng làm việc gợi ý

1. Chuẩn bị dữ liệu bằng `aime2025/prepare_data.py`.
2. Chạy đánh giá nhanh bằng `aime2025/eval_api.py` hoặc `aime2025/eval_api_tool.py`.
3. Nếu cần train/eval Agent0 đầy đủ, dùng các script shell trong `aime2025/`.

## Lưu ý

- Các script trong `aime2025/` có script dành cho:
  - máy local/macOS (đánh giá qua API),
  - server có NVIDIA GPU (train/eval Agent0 đầy đủ).
- Kết quả mặc định được lưu dạng `results_*.json` trong cùng thư mục `Data/aime2025/`.
