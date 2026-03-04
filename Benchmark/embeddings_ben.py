"""
Benchmark embedding models for retrieval tasks with GPU acceleration and accurate memory estimation
"""

import numpy as np
import torch
import gc
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from Embeddings.embeddings import EmbeddingEngine


def detect_long_context_support(model_name: str) -> bool:
    """
    Phát hiện model có support long context không
    
    Args:
        model_name: Tên model
        
    Returns:
        True nếu model support long context
    """
    long_context_models = [
        'llama', 'longformer', 'bigbird', 'long', 'LED', 
        'led-base', 'long-t5', 'gpt-neo', 'gpt-j'
    ]
    return any(model in model_name.lower() for model in long_context_models)


def get_model_dtype_info(model: torch.nn.Module) -> Tuple[str, int]:
    """
    Lấy thông tin dtype của model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple (dtype_name, dtype_size_bytes)
    """
    try:
        first_param = next(model.parameters())
        dtype = first_param.dtype
        
        dtype_map = {
            torch.float32: ('float32', 4),
            torch.float16: ('float16', 2),
            torch.bfloat16: ('bfloat16', 2),
            torch.float64: ('float64', 8),
            torch.int8: ('int8', 1),
            torch.int16: ('int16', 2),
            torch.int32: ('int32', 4),
            torch.int64: ('int64', 8)
        }
        
        return dtype_map.get(dtype, ('unknown', 4))
    except:
        return ('unknown', 4)


def get_model_info(engine: EmbeddingEngine) -> Dict:
    """
    Lấy thông tin model từ EmbeddingEngine với memory estimation chính xác
    
    Args:
        engine: EmbeddingEngine instance
        
    Returns:
        Dict chứa model info chi tiết
    """
    model = engine.model
    
    # 1. Number of parameters
    try:
        num_params = sum(p.numel() for p in model.parameters())
    except Exception:
        num_params = 0
    
    # 2. Get dtype information
    dtype_name, dtype_size = get_model_dtype_info(model)
    
    # 3. Max token length
    max_token = getattr(model, 'max_seq_length', None) or \
                getattr(getattr(model, 'tokenizer', None), 'model_max_length', None) or \
                512
    
    # 4. Embedding dimension - use engine's cached value
    embed_dim = engine.embedding_dim
    
    # 5. Memory usage (chính xác theo dtype)
    mem_used_mb = 0.0
    
    if engine.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            mem_used_mb = mem_allocated / (1024 ** 2)
            mem_reserved_mb = mem_reserved / (1024 ** 2)
        except Exception:
            # Fallback: estimate from params với dtype chính xác
            mem_used_mb = (num_params * dtype_size) / (1024 ** 2)
            mem_reserved_mb = mem_used_mb * 1.1  # Estimate overhead
    else:
        # CPU: estimate from params
        mem_used_mb = (num_params * dtype_size) / (1024 ** 2)
        mem_reserved_mb = mem_used_mb
    
    # 6. OOM and calibration stats
    oom_count = getattr(engine, '_oom_count', 0)
    calibrated = getattr(engine, '_calibrated', False)
    
    return {
        'num_params': num_params,
        'num_params_M': round(num_params / 1e6, 2),
        'max_token': max_token,
        'embed_dim': embed_dim,
        'dtype': dtype_name,
        'dtype_size': dtype_size,
        'mem_used_mb': round(mem_used_mb, 2),
        'mem_reserved_mb': round(mem_reserved_mb, 2),
        'oom_count': oom_count,
        'calibrated': calibrated,
        'supports_long_context': detect_long_context_support(engine.model_name),
        'device': engine.device
    }


def compute_metrics_batch_cpu(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_ids: List[str],
    relevant_mapping: List[set],
    k: int,
    batch_size: int = 100,
    show_progress: bool = True
) -> Dict[str, List[float]]:
    """
    Tính metrics trên CPU (dùng cho dữ liệu nhỏ)
    
    Args:
        query_vecs: Query embeddings (N_queries, dim)
        chunk_vecs: Chunk embeddings (N_chunks, dim)
        chunk_ids: List of chunk IDs
        relevant_mapping: List of sets containing relevant chunk IDs per query
        k: Top-K for metrics
        batch_size: Batch size for similarity computation
        show_progress: Show progress bar
        
    Returns:
        Dict of metric lists
    """
    chunk_id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
    
    all_recalls = []
    all_precisions_1 = []
    all_precisions_k = []
    all_mrrs = []
    all_ndcgs = []
    
    num_queries = len(query_vecs)
    
    iterator = range(0, num_queries, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing metrics (CPU)", unit="batch")
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, num_queries)
        batch_queries = query_vecs[batch_start:batch_end]
        
        # Compute similarity for this batch
        similarity_matrix = np.dot(batch_queries, chunk_vecs.T)
        
        # Process each query in batch
        for i, scores in enumerate(similarity_matrix):
            query_idx = batch_start + i
            relevant_ids = relevant_mapping[query_idx]
            
            if not relevant_ids:
                continue
            
            relevant_indices = [chunk_id_to_idx[rid] for rid in relevant_ids 
                               if rid in chunk_id_to_idx]
            
            if not relevant_indices:
                continue
            
            # Top-K indices
            if len(scores) > k:
                top_k_indices = np.argpartition(scores, -k)[-k:]
                top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            else:
                top_k_indices = np.argsort(scores)[::-1][:k]
            
            top_k_ids = [chunk_ids[idx] for idx in top_k_indices]
            
            # Recall@K
            num_relevant_found = sum(1 for cid in top_k_ids if cid in relevant_ids)
            all_recalls.append(num_relevant_found / len(relevant_ids))
            
            # Precision@1
            all_precisions_1.append(1.0 if top_k_ids[0] in relevant_ids else 0.0)
            
            # Precision@K
            all_precisions_k.append(num_relevant_found / k if k > 0 else 0)
            
            # MRR
            mrr_found = False
            for rank, cid in enumerate(top_k_ids, 1):
                if cid in relevant_ids:
                    all_mrrs.append(1.0 / rank)
                    mrr_found = True
                    break
            if not mrr_found:
                all_mrrs.append(0.0)
            
            # NDCG@K
            relevances = [1 if cid in relevant_ids else 0 for cid in top_k_ids]
            dcg = sum(rel / math.log2(pos + 2) for pos, rel in enumerate(relevances))
            num_ideal = min(k, len(relevant_ids))
            idcg = sum(1.0 / math.log2(pos + 2) for pos in range(num_ideal))
            all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
        # Cleanup batch similarity matrix
        del similarity_matrix
    
    return {
        'recalls': all_recalls,
        'precisions_1': all_precisions_1,
        'precisions_k': all_precisions_k,
        'mrrs': all_mrrs,
        'ndcgs': all_ndcgs
    }


def compute_metrics_batch_gpu(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_ids: List[str],
    relevant_mapping: List[set],
    k: int,
    device: str = "cuda",
    batch_size: int = 100,
    show_progress: bool = True,
    use_mixed_precision: bool = True
) -> Dict[str, List[float]]:
    """
    Tính metrics trên GPU để tăng tốc (dùng cho dữ liệu lớn)
    
    Args:
        query_vecs: Query embeddings (N_queries, dim)
        chunk_vecs: Chunk embeddings (N_chunks, dim)
        chunk_ids: List of chunk IDs
        relevant_mapping: List of sets containing relevant chunk IDs
        k: Top-K for metrics
        device: Device to use ('cuda')
        batch_size: Batch size for GPU processing
        show_progress: Show progress bar
        use_mixed_precision: Use automatic mixed precision
        
    Returns:
        Dict of metric lists
    """
    chunk_id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
    
    # Chuyển chunk vectors sang GPU (chỉ làm 1 lần)
    chunk_tensor = torch.from_numpy(chunk_vecs).to(device)
    if use_mixed_precision and chunk_vecs.dtype == np.float32:
        chunk_tensor = chunk_tensor.half()
    
    all_recalls = []
    all_precisions_1 = []
    all_precisions_k = []
    all_mrrs = []
    all_ndcgs = []
    
    num_queries = len(query_vecs)
    
    iterator = range(0, num_queries, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="🚀 Computing metrics (GPU)", unit="batch")
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, num_queries)
        
        # Chuyển batch queries sang GPU
        batch_queries = torch.from_numpy(query_vecs[batch_start:batch_end]).to(device)
        if use_mixed_precision and query_vecs.dtype == np.float32:
            batch_queries = batch_queries.half()
        
        # GPU matrix multiplication with autocast
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            similarity_matrix = torch.mm(batch_queries, chunk_tensor.T)
        
        # Lấy top-k indices ngay trên GPU (nhanh hơn)
        top_k_values, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
        
        # Chuyển về CPU để xử lý kết quả
        top_k_indices_cpu = top_k_indices.cpu().numpy()
        
        # Process each query
        for i, indices in enumerate(top_k_indices_cpu):
            query_idx = batch_start + i
            relevant_ids = relevant_mapping[query_idx]
            
            if not relevant_ids:
                continue
            
            top_k_ids = [chunk_ids[idx] for idx in indices]
            
            # Tính metrics
            num_relevant_found = sum(1 for cid in top_k_ids if cid in relevant_ids)
            all_recalls.append(num_relevant_found / len(relevant_ids))
            all_precisions_1.append(1.0 if top_k_ids[0] in relevant_ids else 0.0)
            all_precisions_k.append(num_relevant_found / k)
            
            # MRR
            mrr_found = False
            for rank, cid in enumerate(top_k_ids, 1):
                if cid in relevant_ids:
                    all_mrrs.append(1.0 / rank)
                    mrr_found = True
                    break
            if not mrr_found:
                all_mrrs.append(0.0)
            
            # NDCG
            relevances = [1 if cid in relevant_ids else 0 for cid in top_k_ids]
            dcg = sum(rel / math.log2(pos + 2) for pos, rel in enumerate(relevances))
            num_ideal = min(k, len(relevant_ids))
            idcg = sum(1.0 / math.log2(pos + 2) for pos in range(num_ideal))
            all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
        # Periodic cleanup
        del similarity_matrix, top_k_values, top_k_indices, batch_queries
        if batch_start % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    # Final cleanup
    del chunk_tensor
    torch.cuda.empty_cache()
    
    return {
        'recalls': all_recalls,
        'precisions_1': all_precisions_1,
        'precisions_k': all_precisions_k,
        'mrrs': all_mrrs,
        'ndcgs': all_ndcgs
    }


def compute_metrics_hybrid(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_ids: List[str],
    relevant_mapping: List[set],
    k: int,
    device: str = "cuda",
    batch_size: int = 100,
    gpu_threshold: int = 10000,  # Dùng GPU nếu chunk > 10k
    show_progress: bool = True,
    use_mixed_precision: bool = True
) -> Dict[str, List[float]]:
    """
    Tự động chọn phương pháp tính toán tối ưu dựa trên kích thước dữ liệu
    
    Args:
        query_vecs: Query embeddings (N_queries, dim)
        chunk_vecs: Chunk embeddings (N_chunks, dim)
        chunk_ids: List of chunk IDs
        relevant_mapping: List of sets containing relevant chunk IDs
        k: Top-K for metrics
        device: Device to use
        batch_size: Batch size for processing
        gpu_threshold: Số chunks tối thiểu để dùng GPU
        show_progress: Show progress bar
        use_mixed_precision: Use automatic mixed precision for GPU
        
    Returns:
        Dict of metric lists
    """
    use_gpu = (
        device == "cuda" and 
        torch.cuda.is_available() and 
        len(chunk_vecs) > gpu_threshold
    )
    
    if use_gpu:
        print(f"🚀 Using GPU acceleration ({len(chunk_vecs)} chunks)")
        return compute_metrics_batch_gpu(
            query_vecs=query_vecs,
            chunk_vecs=chunk_vecs,
            chunk_ids=chunk_ids,
            relevant_mapping=relevant_mapping,
            k=k,
            device=device,
            batch_size=batch_size,
            show_progress=show_progress,
            use_mixed_precision=use_mixed_precision
        )
    else:
        print(f"💻 Using CPU computation ({len(chunk_vecs)} chunks)")
        return compute_metrics_batch_cpu(
            query_vecs=query_vecs,
            chunk_vecs=chunk_vecs,
            chunk_ids=chunk_ids,
            relevant_mapping=relevant_mapping,
            k=k,
            batch_size=batch_size,
            show_progress=show_progress
        )


def benchmark_embedding(
    chunks: List[Dict],
    samples: List[Dict],
    model_name: str = "BAAI/bge-m3",
    k: int = 10,
    similarity_batch_size: int = 100,
    normalize: bool = True,
    device: str = None,
    auto_batch: bool = True,
    safety_margin: float = 0.8,
    gpu_threshold: int = 10000,
    use_mixed_precision: bool = True
) -> Dict[str, float]:
    """
    Benchmark embedding model với retrieval task
    
    Args:
        chunks: List of dicts with '_id' and 'text' keys
        samples: List of dicts with 'question' and 'relevant_ids' keys
        model_name: HuggingFace model name
        k: Top-K for evaluation
        similarity_batch_size: Batch size for similarity computation
        normalize: Whether to normalize embeddings
        device: Device to use ('cuda' or 'cpu')
        auto_batch: Enable automatic batch sizing
        safety_margin: Safety margin for batch size (0.5-1.0)
        gpu_threshold: Số chunks tối thiểu để dùng GPU cho similarity
        use_mixed_precision: Use mixed precision on GPU
        
    Returns:
        Dict containing all metrics and model info
    """
    
    print(f"\n{'='*80}")
    print(f"🔬 EMBEDDING BENCHMARK: {model_name}")
    print(f"{'='*80}")
    
    # Initialize engine
    print(f"🔄 Initializing EmbeddingEngine...")
    engine = EmbeddingEngine(
        model_name=model_name,
        device=device,
        batch_size=64,
        normalize=normalize,
        auto_batch=auto_batch,
        safety_margin=safety_margin
    )
    
    # ===== GET MODEL INFO =====
    print(f"\n📊 Collecting model information...")
    model_info = get_model_info(engine)
    
    print(f"  • Parameters    : {model_info['num_params_M']}M ({model_info['num_params']:,})")
    print(f"  • Max tokens    : {model_info['max_token']}")
    print(f"  • Embed dim     : {model_info['embed_dim']}")
    print(f"  • Dtype         : {model_info['dtype']} ({model_info['dtype_size']} bytes)")
    print(f"  • Memory used   : {model_info['mem_used_mb']:.2f} MB")
    print(f"  • Memory reserved: {model_info['mem_reserved_mb']:.2f} MB")
    print(f"  • Long context  : {model_info['supports_long_context']}")
    print(f"  • OOM count     : {model_info.get('oom_count', 0)}")
    
    # ===== 1. PREPARE CHUNKS =====
    print(f"\n📦 Processing {len(chunks)} chunks...")
    
    chunks_with_text = []
    chunk_ids = []
    
    for chunk in chunks:
        ch = chunk.copy()
        if "title" in ch and ch.get("title") and str(ch["title"]).strip():
            ch["text"] = f"{ch['title']}: {ch['text']}"
        chunks_with_text.append(ch)
        chunk_ids.append(str(chunk["_id"]))
    
    # ===== 2. ENCODE CHUNKS =====
    print("⚙️  Encoding chunks...")
    time_start = datetime.now()
    embedded_chunks = engine.embed_chunks(chunks_with_text)
    time_end = datetime.now()
    embed_chunks_time = (time_end - time_start).total_seconds()
    
    chunk_vecs = np.array([ch["vector"] for ch in embedded_chunks])
    print(f"✅ Encoded {len(chunk_vecs)} chunks in {embed_chunks_time:.2f}s")
    
    # ===== 3. PREPARE QUERIES =====
    print(f"\n❓ Processing {len(samples)} queries...")
    
    query_texts = [sample["question"] for sample in samples]
    relevant_mapping = [
        {str(rid) for rid in sample["relevant_ids"]} 
        for sample in samples
    ]
    
    # ===== 4. ENCODE QUERIES =====
    print("⚙️  Encoding queries...")
    time_start = datetime.now()
    query_vecs = engine.embed_queries_batch(query_texts)
    time_end = datetime.now()
    embed_queries_time = (time_end - time_start).total_seconds()
    
    print(f"✅ Encoded {len(query_vecs)} queries in {embed_queries_time:.2f}s")
    
    # ===== 5. COMPUTE METRICS =====
    print(f"\n🧮 Computing metrics (k={k})...")
    
    metrics = compute_metrics_hybrid(
        query_vecs=query_vecs,
        chunk_vecs=chunk_vecs,
        chunk_ids=chunk_ids,
        relevant_mapping=relevant_mapping,
        k=k,
        device=engine.device,
        batch_size=similarity_batch_size,
        gpu_threshold=gpu_threshold,
        show_progress=True,
        use_mixed_precision=use_mixed_precision
    )
    
    # ===== 6. AGGREGATE RESULTS =====
    results = {
        # Performance metrics
        "recall@k": np.mean(metrics['recalls']) if metrics['recalls'] else 0.0,
        "precision@1": np.mean(metrics['precisions_1']) if metrics['precisions_1'] else 0.0,
        "precision@k": np.mean(metrics['precisions_k']) if metrics['precisions_k'] else 0.0,
        "mrr": np.mean(metrics['mrrs']) if metrics['mrrs'] else 0.0,
        "ndcg@k": np.mean(metrics['ndcgs']) if metrics['ndcgs'] else 0.0,
        "num_queries_evaluated": len(metrics['recalls']),
        
        # Model info
        "model": model_name,
        "num_params": model_info['num_params'],
        "num_params_M": model_info['num_params_M'],
        "max_token": model_info['max_token'],
        "embed_dim": model_info['embed_dim'],
        "dtype": model_info['dtype'],
        "dtype_size": model_info['dtype_size'],
        "mem_used_mb": model_info['mem_used_mb'],
        "mem_reserved_mb": model_info['mem_reserved_mb'],
        "oom_count": model_info.get('oom_count', 0),
        "calibrated": model_info.get('calibrated', False),
        "supports_long_context": model_info.get('supports_long_context', False),
        
        # Timing
        "embed_chunks_time": round(embed_chunks_time, 2),
        "embed_queries_time": round(embed_queries_time, 2),
        "avg_query_time_ms": round((embed_queries_time / len(samples) * 1000), 2) if samples else 0,
        "total_time": round(embed_chunks_time + embed_queries_time, 2),
        
        # Config
        "k": k,
        "normalize": normalize,
        "batch_size_used": engine.current_batch_size,
        "device": engine.device,
        "similarity_method": "gpu" if len(chunk_vecs) > gpu_threshold and engine.device == "cuda" else "cpu",
        "mixed_precision": use_mixed_precision
    }
    
    # ===== 7. CLEANUP =====
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✅ Benchmark completed!")
    
    return results


def print_benchmark_results(results: Dict, verbose: bool = True):
    """
    In kết quả benchmark với format đẹp
    
    Args:
        results: Dict from benchmark_embedding()
        verbose: Show detailed info
    """
    print("\n" + "="*100)
    print(" "*35 + "📊 EMBEDDING BENCHMARK RESULTS")
    print("="*100)
    
    print(f"\n🤖 MODEL: {results['model']}")
    print("-"*100)
    
    # Model specs
    print("\n📐 MODEL SPECIFICATIONS:")
    print(f"  • Parameters      : {results['num_params_M']}M ({results['num_params']:,})")
    print(f"  • Max tokens      : {results['max_token']}")
    print(f"  • Embedding dim   : {results['embed_dim']}")
    print(f"  • Data type       : {results.get('dtype', 'N/A')} ({results.get('dtype_size', 4)} bytes)")
    print(f"  • Memory used     : {results['mem_used_mb']:.2f} MB")
    print(f"  • Memory reserved : {results.get('mem_reserved_mb', 0):.2f} MB")
    print(f"  • Long context    : {results.get('supports_long_context', False)}")
    
    if verbose:
        print(f"  • Batch size used : {results.get('batch_size_used', 'N/A')}")
        print(f"  • Normalized      : {results.get('normalize', 'N/A')}")
        print(f"  • OOM events      : {results.get('oom_count', 0)}")
        print(f"  • Calibrated      : {results.get('calibrated', False)}")
        print(f"  • Device          : {results.get('device', 'N/A')}")
        print(f"  • Similarity      : {results.get('similarity_method', 'cpu')}")
        print(f"  • Mixed precision : {results.get('mixed_precision', False)}")
    
    # Performance metrics
    print("\n🎯 PERFORMANCE METRICS:")
    print(f"  • Recall@{results['k']:<2}      : {results['recall@k']:.4f}")
    print(f"  • Precision@1      : {results['precision@1']:.4f}")
    print(f"  • Precision@{results['k']:<2}   : {results['precision@k']:.4f}")
    print(f"  • MRR              : {results['mrr']:.4f}")
    print(f"  • NDCG@{results['k']:<2}        : {results['ndcg@k']:.4f}")
    
    # Timing
    print("\n⏱️  TIMING:")
    print(f"  • Embed chunks    : {results['embed_chunks_time']:.2f}s")
    print(f"  • Embed queries   : {results['embed_queries_time']:.2f}s")
    print(f"  • Avg per query   : {results['avg_query_time_ms']:.2f}ms")
    print(f"  • Total time      : {results.get('total_time', 0):.2f}s")
    
    # Stats
    print("\n📈 STATISTICS:")
    print(f"  • Queries evaluated: {results['num_queries_evaluated']}")
    
    print("="*100)


def save_benchmark_results(results: Dict, output_path: str):
    """
    Save benchmark results to JSON file
    
    Args:
        results: Dict from benchmark_embedding()
        output_path: Path to save JSON file
    """
    import json
    
    # Convert numpy types to Python types
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.integer, np.floating)):
            results_serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            results_serializable[k] = v.tolist()
        elif isinstance(v, (np.bool_, bool)):
            results_serializable[k] = bool(v)
        else:
            results_serializable[k] = v
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")


def compare_models(results_list: List[Dict], output_path: str = None):
    """
    So sánh nhiều models
    
    Args:
        results_list: List of result dicts from benchmark_embedding()
        output_path: Optional path to save comparison CSV
    """
    import pandas as pd
    
    print("\n" + "="*100)
    print(" "*35 + "📊 MODEL COMPARISON")
    print("="*100)
    
    # Create comparison dataframe
    comparison = []
    for r in results_list:
        comparison.append({
            'Model': r['model'].split('/')[-1],
            'Params(M)': r['num_params_M'],
            'Dim': r['embed_dim'],
            'VRAM(MB)': r['mem_used_mb'],
            f'R@{r["k"]}': r[f'recall@{r["k"]}'],
            'P@1': r['precision@1'],
            'MRR': r['mrr'],
            f'NDCG@{r["k"]}': r[f'ndcg@{r["k"]}'],
            'Time(s)': r['total_time'],
            'QPS': round(len(r.get('num_queries_evaluated', 0)) / r['total_time'], 2) if r['total_time'] > 0 else 0
        })
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print("="*100)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Comparison saved to: {output_path}")


# # ===== USAGE EXAMPLE =====
# if __name__ == "__main__":
#     # Example usage with realistic data
#     print("🔬 Embedding Benchmark Tool")
#     print("="*60)
    
#     # Create sample data
#     print("\n📁 Creating sample dataset...")
#     chunks = [
#         {
#             "_id": f"doc_{i}", 
#             "text": f"Đây là nội dung của document thứ {i}. " * 20,
#             "title": f"Title {i}" if i % 3 == 0 else None
#         }
#         for i in range(1000)  # 1000 chunks
#     ]
    
#     samples = [
#         {
#             "question": f"Câu hỏi về document {i}",
#             "relevant_ids": [f"doc_{i}"]
#         }
#         for i in range(100)  # 100 queries
#     ]
    
#     # Benchmark multiple models
#     models_to_test = [
#         "BAAI/bge-m3",
#         "BAAI/bge-small-en-v1.5",
#         # Add more models as needed
#     ]
    
#     all_results = []
    
#     for model_name in models_to_test:
#         try:
#             results = benchmark_embedding(
#                 chunks=chunks,
#                 samples=samples,
#                 model_name=model_name,
#                 k=10,
#                 device="cuda",  # Change to "cpu" if no GPU
#                 auto_batch=True,
#                 gpu_threshold=5000,  # Use GPU if >5000 chunks
#                 use_mixed_precision=True
#             )
            
#             print_benchmark_results(results)
#             all_results.append(results)
            
#             # Save individual result
#             save_benchmark_results(
#                 results, 
#                 f"benchmark_{model_name.replace('/', '_')}.json"
#             )
            
#         except Exception as e:
#             print(f"❌ Error benchmarking {model_name}: {e}")
    
#     # Compare all models
#     if len(all_results) > 1:
#         compare_models(all_results, "model_comparison.csv")
    
#     print("\n✅ All benchmarks completed!")