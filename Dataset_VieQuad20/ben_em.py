from Embeddings.embeddings import EmbeddingEngine
from Benchmark.embeddings_ben import benchmark_embedding, print_benchmark_results, save_benchmark_results

import pandas as pd
from pathlib import Path
import gc
import torch
import copy
from datasets import load_dataset
import numpy as np
import os

os.makedirs("File_JSON", exist_ok=True)


model = ["keepitreal/vietnamese-sbert", 
         "intfloat/multilingual-e5-large-instruct", 
         "Qwen/Qwen3-Embedding-0.6B", 
         "intfloat/multilingual-e5-base", 
         "intfloat/multilingual-e5-small",
         "BAAI/bge-m3"]


dataset = load_dataset("taidng/UIT-ViQuAD2.0", cache_dir=".cache/UIT_ViQuAD2")

rng = np.random.default_rng(42)
idx = rng.choice(len(dataset["train"]), 3000, replace=False)

data = dataset["train"].select(sorted(idx))

# Chuyển sang DataFrame ngay từ đầu
df = pd.DataFrame(data)

# --- 1. CORPUS (nhanh gấp 10x) ---
# Lọc bỏ impossible và lấy context duy nhất
df_valid = df[~df['is_impossible']].reset_index(drop=True)

# Tạo mapping context -> id (dùng pd.factorize cực nhanh)
contexts = df_valid['context'].values
context_ids, unique_contexts = pd.factorize(contexts)
context_to_id = {text: f"doc_{i}" for i, text in enumerate(unique_contexts)}

# Tạo corpus DataFrame
corpus_df = pd.DataFrame({
    "_id": [f"doc_{i}" for i in range(len(unique_contexts))],
    "text": unique_contexts
})

# --- 2. QUERIES (vectorized) ---
# Tạo queries DataFrame
queries_df = pd.DataFrame({
    "_id": [f"q_{i}" for i in range(len(df_valid))],
    "text": df_valid['question'].values
})

# --- 3. QRELS (vectorized) ---
# Tạo qrels DataFrame với vectorization
qrels_data = {
    "query-id": [f"q_{i}" for i in range(len(df_valid))],
    "corpus-id": [context_to_id[ctx] for ctx in contexts],
    "score": 1
}
qrels_df = pd.DataFrame(qrels_data)

# --- 4. SAMPLES (nhanh với merge) ---
# Dùng merge thay vì vòng lặp
qrels_grouped = qrels_df.groupby('query-id')['corpus-id'].apply(list).reset_index()
samples_df = pd.merge(
    queries_df, 
    qrels_grouped, 
    left_on='_id', 
    right_on='query-id'
)
samples_df = samples_df.rename(columns={
    'text': 'question',
    'corpus-id': 'relevant_ids'
})
samples = samples_df[['question', 'relevant_ids']].to_dict('records')

# Convert
chunks = corpus_df.to_dict('records')

results = []

for md in model:
  try:
    print(f"\n🚀 Benchmarking: {md}")

    chunks_copy = copy.deepcopy(chunks)

    kq = benchmark_embedding(
        chunks=chunks_copy,
        samples=samples,
        k=5,
        model_name=md
    )

    results.append(kq)

    file_path = os.path.join(
        "File_JSON",
        f"VieQuAD_20_benchmark_{md.replace('/', '_')}.json"
    )

    save_benchmark_results(kq, file_path)

    print_benchmark_results(kq)

    del chunks_copy
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

  except Exception as e:
    print(f"❌ Error benchmarking: {md}")
    print(e)


performance = []
info = []
timing = []

for re in results:
    # 1. Nhóm chỉ số chất lượng (Độ chính xác)
    performance.append({
        "model": re["model"],
        "recall@k": re["recall@k"],
        "precision@k": re["precision@k"],
        "mrr": re["mrr"],
        "ndcg@k": re["ndcg@k"]
    })

    # 2. Nhóm thông số kỹ thuật của Model
    info.append({
        "model": re["model"],
        "num_params_M": re["num_params_M"],
        "max_token": re["max_token"],
        "embed_dim": re["embed_dim"],
        "mem_used_mb": re["mem_used_mb"]
    })

    # 3. Nhóm đo lường thời gian (Tốc độ)
    timing.append({
        "model": re["model"],
        "avg_query_time_ms": re["avg_query_time_ms"],
        "embed_chunks_time": re["embed_chunks_time"],
        "k": re["k"]
    })

print("\n--- PERFORMANCE RANKING ---")
print(pd.DataFrame(performance))

print("\n--- MODEL SPECIFICATIONS ---")
print(pd.DataFrame(info))

print("\n--- MODEL SPECIFICATIONS ---")
print(pd.DataFrame(timing))