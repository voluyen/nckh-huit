from Benchmark.reranking_ben import benchmark_reranking, print_results, export_results

import pandas as pd
from pathlib import Path
import gc
import torch
import copy
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



"""XQuAD Vietnamese - Chuẩn benchmark QA"""
ds = load_dataset("xquad", "xquad.vi", split="validation", cache_dir=".cache/XQuAD")
df = pd.DataFrame(ds)

# --- CORPUS ---
contexts = df['context'].values
context_ids, unique_contexts = pd.factorize(contexts)

corpus_df = pd.DataFrame({
    "_id": [f"doc_{i}" for i in range(len(unique_contexts))],
    "text": unique_contexts
})

# --- QUERIES ---
queries_df = pd.DataFrame({
    "_id": [f"q_{i}" for i in range(len(df))],
    "text": df['question'].values
})

# --- QRELS ---
context_to_id = {text: f"doc_{i}" for i, text in enumerate(unique_contexts)}
qrels_df = pd.DataFrame({
    "query-id": [f"q_{i}" for i in range(len(df))],
    "corpus-id": [context_to_id[ctx] for ctx in contexts],
    "score": 1
})

# --- SAMPLES ---
qrels_grouped = qrels_df.groupby('query-id')['corpus-id'].apply(list).reset_index()
samples_df = pd.merge(queries_df, qrels_grouped, left_on='_id', right_on='query-id')
samples_df = samples_df.rename(columns={'text': 'question', 'corpus-id': 'relevant_ids'})
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
            ten = f"ben_rerank_{clean_md}_XQuad_VI.csv"
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
