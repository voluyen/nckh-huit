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

"""Ben rerank for ds VieQuad"""

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
    