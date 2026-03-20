"""
Download AIME2025 dataset from HuggingFace and convert to parquet format
compatible with Agent0's data loading pipeline.
"""
from datasets import load_dataset, concatenate_datasets
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# Load both subsets: AIME2025-I and AIME2025-II
aime1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
aime2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

# Combine all 30 problems
full_dataset = concatenate_datasets([aime1, aime2])

# Agent0 executor_train expects 'prompt' key by default,
# but can be configured via data.prompt_key
# Keep original field names: 'question' and 'answer'
# We'll configure Agent0 to use prompt_key=question, answer_key=answer

# Save as parquet
test_path = os.path.join(output_dir, "aime2025_test.parquet")
full_dataset.to_parquet(test_path)
print(f"Saved {len(full_dataset)} examples to {test_path}")

# Also save each subset separately
aime1_path = os.path.join(output_dir, "aime2025_I_test.parquet")
aime2_path = os.path.join(output_dir, "aime2025_II_test.parquet")
aime1.to_parquet(aime1_path)
aime2.to_parquet(aime2_path)
print(f"Saved AIME2025-I ({len(aime1)} examples) to {aime1_path}")
print(f"Saved AIME2025-II ({len(aime2)} examples) to {aime2_path}")
