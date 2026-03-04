from Embeddings.embeddings import EmbeddingEngine
from Benchmark.embeddings_ben import benchmark_embedding, print_benchmark_results, save_benchmark_results

import pandas as pd
from pathlib import Path
import gc
import torch
import copy
import os

os.makedirs("File_JSON", exist_ok=True)


model = ["keepitreal/vietnamese-sbert", 
         "intfloat/multilingual-e5-large-instruct", 
         "Qwen/Qwen3-Embedding-0.6B", 
         "intfloat/multilingual-e5-base", 
         "intfloat/multilingual-e5-small",
         "BAAI/bge-m3"]


from huggingface_hub import hf_hub_download

repo = "mteb/VieQuADRetrieval"

corpus_path = hf_hub_download(repo, "corpus/validation-00000-of-00001.parquet", repo_type="dataset")
queries_path = hf_hub_download(repo, "queries/validation-00000-of-00001.parquet", repo_type="dataset")
qrels_path = hf_hub_download(repo, "qrels/validation-00000-of-00001.parquet", repo_type="dataset")

corpus = pd.read_parquet(corpus_path)
queries = pd.read_parquet(queries_path)
qrels = pd.read_parquet(qrels_path)


chunks = corpus.to_dict(orient="records")

qs = queries.to_dict(orient="records")

da = qrels.to_dict(orient="records")

chunks = [
    x for x in chunks
    if not any(x != y and x["text"] in y["text"] for y in chunks)
]

da = [
    x for x in da
    if any(x["corpus-id"] == y["_id"] for y in chunks)
]

samples = []

for que in qs:
    relevant_ids = {
        x["corpus-id"]
        for x in da
        if x["query-id"] == que["_id"]
    }

    if not relevant_ids:
        continue

    samples.append({
        "question": que["text"],
        "relevant_ids": relevant_ids
    })


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
        f"VieQuAD_benchmark_{md.replace('/', '_')}.json"
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