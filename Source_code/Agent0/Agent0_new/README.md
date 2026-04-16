# Agent0_new — Self-Evolving Code Knowledge Base + Benchmark

Hệ thống tự tạo knowledge base (KB) bằng LLM, sau đó dùng KB đó để giải bài code mới qua few-shot RAG. Đánh giá trên MBPP / MBPP+ benchmark.

## Kiến trúc

```
curriculum_planner.py      Phase 1: LLM tự plan subtopics + reflection
         |
run_agent0_mbpp_curriculum.py   Phase 2: Sinh tasks, verify, repair -> Knowledge Base
         |
    executor.py            Generate code, verify, diagnose, repair (code + tests)
         |
knowledge_retriever.py     Vector DB: embed prompts, query similar, format few-shot
         |
  benchmark_mbpp.py        Benchmark trên MBPP/MBPP+ (EvalPlus): with KB vs baseline
```

## Cài đặt

```bash
cd Source_code/Agent0/Agent0_new
pip install -r requirements_agent0_lite.txt
pip install evalplus   # cho benchmark
```

### API Keys

Tạo file `.env` ở thư mục gốc project:

```bash
# .env
GROQ_API_KEY="gsk_..."           # Groq (miễn phí, rate limit 6000 TPM)
OLLAMA_API_KEY="..."              # Ollama Cloud (không rate limit)
GEMINI_API_KEY="..."              # (optional)
```

### Provider hỗ trợ

| Provider | Model ví dụ | Rate limit | Ghi chú |
|---|---|---|---|
| `ollama` | `qwen3:4b`, `llama3.1:8b` | Không | Local, cần cài Ollama |
| `ollama-cloud` | `gemma3:4b`, `gemma3:12b` | Không | Cloud, cần `OLLAMA_API_KEY` |
| `groq` | `llama-3.3-70b-versatile`, `qwen/qwen3-32b` | 6000 TPM | Nhanh nhưng limit |
| `openai` | `gpt-4o-mini` | Theo plan | Cần `OPENAI_API_KEY` |
| `offline` | — | — | Không gọi LLM |

### Embedding model

Cần **Ollama local** chạy model `mxbai-embed-large` cho vector search:

```bash
ollama pull mxbai-embed-large
```

---

## 1. Tạo Knowledge Base

### Bước 1: Plan subtopics (optional, xem plan trước)

```bash
python curriculum_planner.py \
  --domain "coding" \
  --total_tasks 30 \
  --reflection_rounds 2 \
  --provider ollama-cloud \
  --model "gemma3:4b" \
  --output logs/plan.json
```

LLM sẽ:
1. Tự nghĩ ra các subtopics (arrays, sorting, DP, graphs, ...)
2. Reflection: review plan, thêm topics thiếu (recursion, backtracking, sliding window, ...)
3. Output plan JSON

### Bước 2: Sinh KB tự động

```bash
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud \
  --model "gemma3:4b" \
  --synthetic_only \
  --synthetic_generator llm \
  --synthetic_count 30 \
  --synthetic_domain "coding" \
  --strategy all \
  --items_per_strategy 10 \
  --log_file logs/kb_run.log
```

Pipeline cho mỗi task:
```
Plan subtopics -> Generate task + tests -> Generate code -> Verify
  -> FAIL? -> Diagnose root cause -> Repair code (x2)
    -> Still FAIL? -> Judge: CODE or TEST wrong?
      -> TEST wrong -> Fix tests (x2)
      -> CODE wrong -> Repair code 1 more time
  -> PASS -> Accept vào KB
```

### Chạy thêm nhiều batch (KB tích lũy)

Mỗi lần chạy sẽ **append** vào KB hiện có, tự skip bài trùng:

```bash
# Batch 1: coding chung
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 30 --synthetic_domain "coding" \
  --strategy all --items_per_strategy 10

# Batch 2: DP + recursion (đổi seed để đa dạng)
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 20 --synthetic_domain "dynamic programming, recursion, memoization" \
  --strategy all --items_per_strategy 7 --seed 200

# Batch 3: string + data structures
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 20 --synthetic_domain "string manipulation, regex, hash tables, sets" \
  --strategy all --items_per_strategy 7 --seed 300

# Batch 4: sorting + searching
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 20 --synthetic_domain "sorting, searching, binary search, two pointers" \
  --strategy all --items_per_strategy 7 --seed 400

# Batch 5: math
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 20 --synthetic_domain "math, number theory, combinatorics, probability" \
  --strategy all --items_per_strategy 7 --seed 500
```

### Output

KB lưu theo model:

```
Data/mbpp/curriculum_outputs/
└── gemma3_4b/
    ├── knowledge_base.jsonl        # KB chính (tích lũy qua mỗi batch)
    ├── knowledge_base.index.json   # Vector index (tự cập nhật)
    ├── rejected.jsonl              # Bài bị reject
    └── summary.json                # Thống kê lần chạy cuối
```

### Kiểm tra KB

```bash
python -c "
import json
entries = [json.loads(l) for l in open('Data/mbpp/curriculum_outputs/gemma3_4b/knowledge_base.jsonl') if l.strip()]
print(f'Total: {len(entries)} entries')
print(f'Unique: {len(set(e[\"original_prompt\"] for e in entries))}')
from collections import Counter
topics = Counter(t for e in entries for t in e.get('taxonomy', []))
for t, c in topics.most_common(): print(f'  {t}: {c}')
"
```

---

## 2. Query Knowledge Base

```bash
python knowledge_retriever.py \
  --kb_path Data/mbpp/curriculum_outputs/gemma3_4b/knowledge_base.jsonl \
  --query "Write a function to sort a list using merge sort" \
  --n 3
```

Output: top 3 bài tương tự nhất (cosine similarity) kèm solution code.

### Dùng trong code

```python
from knowledge_retriever import KnowledgeRetriever

retriever = KnowledgeRetriever(kb_path="path/to/knowledge_base.jsonl")
retriever.build_index()  # lần đầu embed, sau đó load từ cache

examples = retriever.query("Write a function to find GCD", n=3)
few_shot = retriever.format_few_shot(examples)
# -> formatted few-shot text cho LLM prompt
```

---

## 3. Giải bài mới bằng KB (Few-shot RAG)

```python
from executor import solve_with_knowledge, RuntimeConfig
from knowledge_retriever import KnowledgeRetriever

# Load KB
retriever = KnowledgeRetriever(kb_path="path/to/knowledge_base.jsonl")
retriever.build_index()

# Task mới cần giải
task = {
    "task_id": "new_1",
    "prompt": "Write a function to find the minimum cost path...",
    "test_list": ["assert min_cost([[1,2],[3,4]], 1, 1) == 5"],
    "challenge_test_list": [],
    "test_setup_code": "",
}

# Query KB -> few-shot
examples = retriever.query(task["prompt"], n=3)
few_shot = retriever.format_few_shot(examples)

# Solve
rt = RuntimeConfig(provider="ollama-cloud", model="gemma3:4b",
                   backend="Ollama Cloud", base_url="https://ollama.com",
                   api_key="your_key")

result = solve_with_knowledge(rt, task, few_shot, repair_rounds=2)
print(f"Accepted: {result.accepted}")
print(result.solution_code)
```

---

## 4. Benchmark trên MBPP / MBPP+

### Chạy full benchmark (so sánh with KB vs baseline)

```bash
python benchmark_mbpp.py run \
  --provider ollama-cloud \
  --model "gemma3:4b" \
  --limit 50 \
  --n_examples 3
```

Sẽ:
1. Generate solutions cho 50 tasks **WITH KB** (few-shot RAG)
2. Generate solutions cho 50 tasks **WITHOUT KB** (baseline)
3. Evaluate cả hai trên MBPP (base tests) và MBPP+ (extended tests)
4. In kết quả pass@1

### Chỉ generate

```bash
# With KB
python benchmark_mbpp.py generate \
  --provider ollama-cloud --model "gemma3:4b" \
  --mode with_kb --limit 100

# Baseline
python benchmark_mbpp.py generate \
  --provider ollama-cloud --model "gemma3:4b" \
  --mode baseline --limit 100
```

### Chỉ evaluate

```bash
python benchmark_mbpp.py evaluate \
  --samples_file Data/mbpp/benchmark_results/gemma3_4b/with_kb_samples.jsonl
```

### Output benchmark

```
Data/mbpp/benchmark_results/
└── gemma3_4b/
    ├── with_kb_samples.jsonl       # Solutions với KB few-shot
    ├── baseline_samples.jsonl      # Solutions không có KB
    └── benchmark_summary.json      # Kết quả pass@1
```

### Kết quả mẫu (20 tasks, gemma3:4b)

| | MBPP (base) | MBPP+ |
|---|---|---|
| **With KB** | **95.0%** | **50.0%** |
| **Baseline** | 85.0% | 45.0% |
| **Cải thiện** | +10% | +5% |

---

## 5. Ghi log chi tiết

```bash
# Log LLM input/output
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud --model "gemma3:4b" \
  --synthetic_only --synthetic_generator llm \
  --synthetic_count 10 --synthetic_domain "coding" \
  --strategy all --items_per_strategy 5 \
  --log_file logs/debug.log \
  --log_level INFO \
  --log_llm_io
```

Log chứa: timestamps, diagnosis, repair attempts, judge verdicts, LLM input/output.

---

## Tham số chính

### run_agent0_mbpp_curriculum.py

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--provider` | `auto` | `ollama`, `ollama-cloud`, `groq`, `openai`, `offline` |
| `--model` | `llama-3.3-70b-versatile` | Tên model |
| `--synthetic_only` | `False` | Không dùng data gốc, tự sinh task |
| `--synthetic_generator` | `auto` | `llm` (LLM sinh) hoặc `template` |
| `--synthetic_count` | `60` | Số task cần sinh |
| `--synthetic_domain` | `coding` | Chủ đề để LLM plan |
| `--strategy` | `all` | `easy_medium_hard`, `diversity`, `mutate`, `all` |
| `--items_per_strategy` | `10` | Số bài mỗi strategy |
| `--repair_rounds` | `2` | Số lần repair tối đa |
| `--seed` | `42` | Random seed (đổi để đa dạng hóa) |
| `--log_llm_io` | `False` | Log LLM input/output |

### benchmark_mbpp.py

| Tham số | Mô tả |
|---|---|
| `--limit N` | Chỉ benchmark N tasks đầu tiên |
| `--n_examples 3` | Số KB examples cho few-shot |
| `--kb_path` | Path đến knowledge_base.jsonl |

---

## Flow tổng quan

```
                    ┌─────────────────────┐
                    │  curriculum_planner  │
                    │  Plan subtopics +   │
                    │  Reflection rounds  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  run_agent0_mbpp_   │
                    │  curriculum.py      │
                    │  Sinh tasks -> KB   │
                    └─────────┬───────────┘
                              │
              ┌───────────────▼───────────────┐
              │         executor.py            │
              │  Generate -> Verify -> Repair  │
              │  Diagnose -> Judge -> Fix      │
              └───────────────┬───────────────┘
                              │ accepted
                    ┌─────────▼───────────┐
                    │  knowledge_base.jsonl│
                    │  + .index.json      │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │ knowledge_retriever  │
                    │ Embed + Vector Search│
                    └─────────┬───────────┘
                              │ top-3 similar
                    ┌─────────▼───────────┐
                    │  benchmark_mbpp.py   │
                    │  MBPP / MBPP+ eval   │
                    │  With KB vs Baseline │
                    └─────────────────────┘
```
