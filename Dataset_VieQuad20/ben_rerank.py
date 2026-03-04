import os
os.environ['HF_HUB_OFFLINE'] = '0'  # PHẢI ĐẶT TRƯỚC KHI IMPORT sentence_transformers
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from Benchmark.reranking_ben import benchmark_reranking, print_results, export_results

import pandas as pd
from pathlib import Path
import gc
import torch
import copy
import numpy as np
from datasets import load_dataset

model_embed = [
    "intfloat/multilingual-e5-large-instruct", 
    "intfloat/multilingual-e5-base", 
    "intfloat/multilingual-e5-small",
    "BAAI/bge-m3"
]
models = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",                
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "BAAI/bge-reranker-v2-m3",          
]


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

ben = {}
for md in model_embed:
    try:
        print("  ")
        print("#####"*15)
        print("  ")
            
        # Benchmark RERANKERS ONLY
        ben[md] = benchmark_reranking(
            chunks=chunks,
            samples=samples,
            embedding_model=md,
            reranker_models=models,
            retrieval_top_k=50,  
            final_k=5,   
            device="cuda",
            batch_size=512
        )

        print_results(results=ben[md], k=5)

        try:
            clean_md = md.replace('/', '_')
            ten = f"ben_rerank_{clean_md}_ViQuad.csv"
            export_results(ben[md], ten)

        except Exception as e:
            print(f"❌ Error saving to Drive: {e}")
        
    except Exception as e:
        print(f"❌ Error benchmarking: {md}")
        print(e)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    print(f"✅ Finished and cleared VRAM for Embedding Model: {md}")



for model_name, results in ben.items():
    print("="*99)
    print(f"Model: {model_name}")
    print_results(results=results, k=5)
    