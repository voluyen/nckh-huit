import os
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import torch
import gc
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import psutil
import warnings
warnings.filterwarnings('ignore')

from Embeddings.embeddings import EmbeddingEngine
from ReRanking.reranker_v2 import Reranker

class AdaptiveReranker:
    """
    🚀 HYBRID RERANKER: Tối ưu cho cả model cũ và mới
    - Tự động phát hiện long context models
    - Fix lỗi 514 tokens của BERT/RoBERTa
    - Batch size tự động điều chỉnh
    - OOM recovery thông minh
    """
    
    # Danh sách models hỗ trợ long context
    LONG_CONTEXT_MODELS = ['llama', 'longformer', 'led', 'bigbird', 'long-t5', 'gpt-neo']
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        safety_margin: float = 0.9,
        max_length: Optional[int] = None,
        vram_threshold: int = 300,  # MB, threshold cho batch calibration
        verbose: bool = True,
        unleash: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.safety_margin = safety_margin
        self.unleash = unleash
        self.vram_threshold = vram_threshold
        
        if self.verbose:
            print(f"🔧 Loading model: {model_name}...")
            if unleash:
                print(f"   ⚡ UNLEASHED MODE: Giữ nguyên khả năng gốc")
        
        self._mem_before = self._get_memory_usage()
        self.model_wrapper = Reranker(model_name, device=self.device)
        self._mem_after = self._get_memory_usage()

        # Phát hiện max_length an toàn
        self._detect_max_length_safe(max_length)
        
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size
        self._calibrated = False
        self._oom_count = 0
        self._model_info = self._get_model_info_auto()

        if self.verbose:
            self._print_summary()

    def _detect_max_length_safe(self, user_max_length: Optional[int]):
        """
        Phát hiện max_length an toàn, fix lỗi 514 của BERT/RoBERTa
        """
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        
        # Lấy từ tokenizer
        tok_max = getattr(tokenizer, 'model_max_length', 512)
        if tok_max > 1e6: 
            tok_max = 512

        # Lấy từ config
        model_max = 512
        if hasattr(model, 'config'):
            for attr in ['max_position_embeddings', 'n_positions', 'max_length']:
                val = getattr(model.config, attr, None)
                if isinstance(val, int) and val > 0:
                    # FIX 514 -> 512 cho BERT/RoBERTa
                    if val == 514: 
                        val = 512
                    model_max = val
                    break
        
        # Native max là min của tokenizer và config
        self._native_max = min(tok_max, model_max)
        
        # Long context models được giữ nguyên
        if self._is_long_context_model():
            self._native_max = max(self._native_max, 4096)
        
        # BGE reranker cũ giới hạn 512
        if "bge-reranker" in self.model_name.lower() and "v2" not in self.model_name.lower():
            self._native_max = min(self._native_max, 512)

        # Quyết định final max_length
        if user_max_length is None:
            self.max_length = self._native_max
        else:
            self.max_length = min(user_max_length, self._native_max)

        # if not self.unleash:
        #     self.max_length = min(self.max_length, 512) # Chế độ an toàn
        
        # Update tokenizer
        tokenizer.model_max_length = self.max_length

    def _is_long_context_model(self) -> bool:
        """
        Kiểm tra model có hỗ trợ long context không.
        Kết hợp giữa check tên và check giới hạn token thực tế.
        """
        # 1. Check theo danh sách tên model (cái cũ của mày)
        name_match = any(m in self.model_name.lower() for m in self.LONG_CONTEXT_MODELS)
        
        # 2. Check theo Native Max (Nếu > 512 thì chắc chắn là Long Context)
        # Chúng ta dùng getattr để tránh lỗi nếu hàm này bị gọi quá sớm
        limit_match = getattr(self, '_native_max', 0) > 512
        
        return name_match or limit_match
    
    def _get_memory_usage(self) -> Dict:
        """Đo memory usage chi tiết"""
        mem = {'ram_mb': psutil.Process().memory_info().rss / (1024**2)}
        if self.device == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            mem['vram_free_mb'] = free / (1024**2)
            mem['vram_allocated'] = torch.cuda.memory_allocated() / (1024**2)
            mem['vram_total_mb'] = total / (1024**2)
        return mem

    def _find_optimal_batch_size(self, query: str, texts: List[str]) -> int:
        """
        Tìm batch size tối ưu bằng binary search
        """
        if self.device != "cuda": 
            return self.current_batch_size
        
        if self.verbose:
            print(f"   🔍 Calibrating batch size...")
        
        # Lấy mẫu dài nhất để test
        test_samples = sorted(texts, key=len, reverse=True)[:3]
        if not test_samples:
            test_samples = ["Sample text for calibration"]
        
        low, high = 1, min(self.initial_batch_size, 1024)
        optimal = 1
        
        while low <= high:
            mid = (low + high) // 2
            try:
                # Tạo batch test
                repeat = mid // len(test_samples) + 1
                test_batch = [(query, t) for t in (test_samples * repeat)[:mid]]
                
                inputs = self.model_wrapper.tokenizer(
                    test_batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.model_wrapper.model(**inputs)
                
                # Kiểm tra VRAM còn lại
                vram_free = self._get_memory_usage().get('vram_free_mb', 0)
                if vram_free > self.vram_threshold:
                    optimal = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            except Exception as e:
                if self.verbose:
                    print(f"      ⚠️ Batch {mid} failed: {str(e)[:50]}")
                high = mid - 1
            finally:
                torch.cuda.empty_cache()
        
        final_batch = max(1, int(optimal * self.safety_margin))
        
        if self.verbose:
            print(f"   ✅ Optimal batch: {final_batch}")
        
        return final_batch

    def rerank_batch(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank với OOM recovery thông minh
        """
        if not candidates:
            return []
        
        texts = [c["text"] for c in candidates]
        n_samples = len(texts)
        scores = np.zeros(n_samples, dtype=np.float32)
        
        # Calibration nếu cần
        if not self._calibrated:
            self.current_batch_size = self._find_optimal_batch_size(query, texts)
            self._calibrated = True
        
        i = 0
        current_bs = self.current_batch_size
        
        while i < n_samples:
            try:
                end = min(i + current_bs, n_samples)
                
                # Tokenize với truncation để an toàn
                inputs = self.model_wrapper.tokenizer(
                    [(query, t) for t in texts[i:end]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model_wrapper.model(**inputs)
                    logits = outputs.logits
                    
                    # Xử lý linh hoạt shape của logits
                    if logits.shape[-1] == 1:
                        batch_scores = logits.view(-1)
                    elif logits.shape[1] > 1:
                        batch_scores = logits[:, 1]  # Class 1 thường là relevant
                    else:
                        batch_scores = logits[:, 0]
                    
                    scores[i:end] = batch_scores.float().cpu().numpy()
                
                i = end
                
                # Periodic cleanup
                if i % (current_bs * 4) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._oom_count += 1
                    
                    if self.verbose:
                        print(f"\n   ⚠️ OOM at batch {current_bs}, reducing...")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if current_bs > 1:
                        current_bs //= 2
                    else:
                        # BS=1 vẫn OOM -> skip sample này
                        if self.verbose:
                            print(f"      🚨 Sample too long, assigning low score")
                        scores[i] = -1e9
                        i += 1
                        current_bs = self.current_batch_size  # Reset
                else:
                    # Lỗi nghiêm trọng khác
                    if self.verbose:
                        print(f"\n   🔥 CUDA Error: {e}")
                    raise e
        
        # Update batch size cho lần sau
        self.current_batch_size = current_bs
        
        # Lấy top-k indices
        if n_samples > top_k * 2:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > -1e8:  # Chỉ lấy sample thành công
                item = candidates[idx].copy()
                item['reranker_score'] = float(scores[idx])
                results.append(item)
        
        return results

    def _get_model_info_auto(self) -> Dict:
        """Lấy thông tin model chi tiết"""
        model = self.model_wrapper.model
        config = getattr(model, 'config', None)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Get embedding dimension
        embed_dim = 'N/A'
        if config:
            for attr in ['hidden_size', 'd_model', 'dim', 'n_embd', 'embedding_dim']:
                if hasattr(config, attr):
                    embed_dim = getattr(config, attr)
                    break
        
        # Calculate memory used
        vram_used = self._mem_after.get('vram_allocated', 0) - self._mem_before.get('vram_allocated', 0)
        vram_free = self._get_memory_usage().get('vram_free_mb', 0)
        
        return {
            'num_params_M': round(num_params / 1e6, 2),
            'max_token': self.max_length,
            'native_max': self._native_max,
            'embed_dim': embed_dim,
            'mem_used_mb': round(vram_used, 2),
            'vram_free_mb': round(vram_free, 2),
            'model_type': getattr(config, 'model_type', 'unknown') if config else 'unknown',
            'long_context': self._is_long_context_model()
        }

    def _print_summary(self):
        """In thông tin tóm tắt"""
        print(f"✅ Ready: {self.model_name.split('/')[-1]}")
        print(f"   📏 Max Length: {self.max_length} (Native: {self._native_max})")
        print(f"   📊 Embed Dim: {self._model_info['embed_dim']}")
        print(f"   🔢 Params: {self._model_info['num_params_M']}M")
        print(f"   💾 VRAM Used: {self._model_info['mem_used_mb']} MB")
        print(f"   💾 VRAM Free: {self._model_info['vram_free_mb']} MB")
        print(f"   📦 Batch Size: {self.current_batch_size}")
        print(f"   🦕 Long Context: {self._model_info['long_context']}")

    def get_model_info(self) -> Dict:
        return self._model_info
    
    def get_stats(self) -> Dict:
        return {
            'model': self.model_name.split('/')[-1],
            'batch_size': self.current_batch_size,
            'oom_count': self._oom_count,
            'calibrated': self._calibrated
        }


def _calculate_complete_metrics(eval_data, k, model_name, model_info, duration, num_samples):
    """
    Tính metrics đầy đủ và chính xác
    """
    all_recalls, all_mrrs, all_ndcgs, all_p1 = [], [], [], []
    
    for item in eval_data:
        top_k_ids = item['top_k_ids'][:k]
        rel_ids = item['relevant_ids']
        
        if not rel_ids:
            continue
        
        relevance = [1 if d in rel_ids else 0 for d in top_k_ids]
        
        # Recall@K
        all_recalls.append(sum(relevance) / len(rel_ids))
        
        # Precision@1
        all_p1.append(relevance[0] if relevance else 0)
        
        # MRR
        mrr = 0
        for i, r in enumerate(relevance, 1):
            if r:
                mrr = 1 / i
                break
        all_mrrs.append(mrr)
        
        # NDCG@K
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(relevance, 1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel_ids), k) + 1))
        all_ndcgs.append(dcg / idcg if idcg > 0 else 0)

    n_valid = len(all_recalls)
    
    return {
        f"recall@{k}": round(np.mean(all_recalls), 4) if all_recalls else 0,
        "precision@1": round(np.mean(all_p1), 4) if all_p1 else 0,
        "mrr": round(np.mean(all_mrrs), 4) if all_mrrs else 0,
        f"ndcg@{k}": round(np.mean(all_ndcgs), 4) if all_ndcgs else 0,
        "num_queries": n_valid,
        "model": model_name.split('/')[-1],
        "model_full": model_name,
        "num_params_M": model_info.get('num_params_M', 0),
        "max_token": model_info.get('max_token', 'N/A'),
        "native_max": model_info.get('native_max', 'N/A'),
        "embed_dim": model_info.get('embed_dim', 'N/A'),
        "mem_used_mb": model_info.get('mem_used_mb', 0),
        "vram_free_mb": model_info.get('vram_free_mb', 0),
        "model_type": model_info.get('model_type', 'unknown'),
        "long_context": model_info.get('long_context', False),
        "rerank_time_sec": round(duration, 2),
        "qps": round(num_samples / duration, 2) if duration > 0 else 0,
        "avg_query_time_ms": round((duration / num_samples * 1000), 2) if num_samples > 0 else 0,
        "k": k
    }


def benchmark_reranking(
    chunks: List[dict],
    samples: List[dict],
    embedding_model: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    reranker_models: List[str] = None,
    retrieval_top_k: int = 50,
    final_k: int = 10,
    batch_size: int = 16,
    verbose: bool = True
) -> List[Dict]:
    """
    Benchmark reranker với HYBRID mode
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 HYBRID RERANKING BENCHMARK")
        print(f"{'='*60}")
        print(f"📦 Embedding: {embedding_model}")
    
    # 1. Embedding phase
    engine = EmbeddingEngine(model_name=embedding_model, device=device, auto_batch=True)
    
    # Encode chunks
    chunk_ids = [str(c["_id"]) for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    embedded = engine.embed_chunks([{"text": t} for t in chunk_texts])
    chunk_vecs = np.array([e["vector"] for e in embedded])
    
    # Encode queries
    query_texts = [s["question"] for s in samples]
    query_vecs = engine.embed_queries_batch(query_texts)
    
    # 2. Retrieval phase
    if verbose:
        print(f"🔍 Retrieving top-{retrieval_top_k}...")
    
    similarity = np.dot(query_vecs, chunk_vecs.T)
    candidates = []
    
    for i in range(len(samples)):
        top_indices = np.argsort(similarity[i])[::-1][:retrieval_top_k]
        candidates.append({
            'query': query_texts[i],
            'rel': {str(rid) for rid in samples[i]["relevant_ids"]},
            'cands': [
                {'doc_id': chunk_ids[idx], 'text': chunk_texts[idx]} 
                for idx in top_indices
            ]
        })
    
    # Cleanup embedding
    del engine, chunk_vecs, query_vecs, similarity
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. Reranking phase
    all_results = []
    
    for model_name in reranker_models:
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Testing: {model_name}")
        
        reranker = AdaptiveReranker(
            model_name=model_name,
            device=device,
            batch_size = batch_size,
            verbose=verbose
        )
        
        start_time = datetime.now()
        eval_data = []
        
        for item in tqdm(candidates, desc=f"  Reranking", disable=not verbose):
            reranked = reranker.rerank_batch(
                item['query'], 
                item['cands'], 
                top_k=final_k
            )
            eval_data.append({
                'top_k_ids': [d['doc_id'] for d in reranked],
                'relevant_ids': item['rel']
            })
        
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics = _calculate_complete_metrics(
            eval_data, final_k, model_name, 
            reranker.get_model_info(), duration, len(samples)
        )
        
        metrics['oom_count'] = reranker._oom_count
        metrics['batch_size_used'] = reranker.current_batch_size
        
        all_results.append(metrics)
        
        if verbose:
            print(f"   ✅ Done: {duration:.2f}s | P@1: {metrics['precision@1']:.4f}")
        
        del reranker, eval_data
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results


def print_results(results: List[Dict], k: int = 10):
    """
    In kết quả benchmark đẹp mắt
    """
    if not results:
        print("❌ No results to display")
        return
    
    print("\n" + "="*120)
    print(f"🏆 HYBRID RERANKING BENCHMARK RESULTS (Top-{k})")
    print("="*120)
    
    # Bảng 1: Performance Metrics
    perf_cols = ['model', 'precision@1', f'recall@{k}', 'mrr', f'ndcg@{k}', 
                 'qps', 'oom_count', 'batch_size_used']
    
    perf_df = pd.DataFrame(results)[perf_cols]
    perf_df.columns = ['Model', 'P@1', f'R@{k}', 'MRR', f'NDCG@{k}', 
                      'QPS', 'OOM', 'Batch']
    
    print("\n📊 PERFORMANCE METRICS:")
    print(perf_df.to_string(index=False))
    
    # Bảng 2: Model Info
    info_cols = ['model', 'num_params_M', 'native_max', 'max_token', 
                 'embed_dim', 'mem_used_mb', 'model_type', 'long_context']
    
    info_df = pd.DataFrame(results)[info_cols]
    info_df.columns = ['Model', 'Params(M)', 'Native Max', 'Used Max', 
                      'Dim', 'VRAM(MB)', 'Type', 'LongCtx']
    
    print("\n📊 MODEL ARCHITECTURE:")
    print(info_df.to_string(index=False))
    
    # Bảng 3: Timing
    time_cols = ['model', 'rerank_time_sec', 'avg_query_time_ms', 'qps']
    time_df = pd.DataFrame(results)[time_cols]
    time_df.columns = ['Model', 'Rerank(s)', 'Avg Q(ms)', 'QPS']
    
    print("\n⏱️  TIMING:")
    print(time_df.to_string(index=False))
    
    # Summary
    print("\n" + "="*120)
    print("📈 SUMMARY STATISTICS:")
    
    best_p1 = max(results, key=lambda x: x['precision@1'])
    best_ndcg = max(results, key=lambda x: x[f'ndcg@{k}'])
    fastest = min(results, key=lambda x: x['rerank_time_sec'])
    
    print(f"   • Best Precision@1: {best_p1['model']} ({best_p1['precision@1']:.4f})")
    print(f"   • Best NDCG@{k}: {best_ndcg['model']} ({best_ndcg[f'ndcg@{k}']:.4f})")
    print(f"   • Fastest: {fastest['model']} ({fastest['rerank_time_sec']:.2f}s)")
    
    total_oom = sum(r.get('oom_count', 0) for r in results)
    if total_oom > 0:
        print(f"   • Total OOM events: {total_oom}")
    
    # Check UNLEASHED models
    unleashed = [r for r in results if r['native_max'] == r['max_token']]
    if unleashed:
        print(f"   • UNLEASHED models: {len(unleashed)}")
    
    print("="*120)


def export_results(results: List[Dict], filename: str = "hybrid_results.csv"):
    """Export kết quả ra file CSV"""
    if not results:
        print("❌ No results to export")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"✅ Results exported to {filename}")

