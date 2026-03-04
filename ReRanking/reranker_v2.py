import os
# os.environ['HF_HUB_OFFLINE'] = '1'  # PHẢI ĐẶT TRƯỚC KHI IMPORT sentence_transformers
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_name: str, device: str = None):
        # Tự động chọn CUDA nếu có
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Loading Reranker: {model_name} on {self.device}")
        
        # trust_remote_code=True là "chìa khóa" để chạy Jina, GTE, Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Dùng torch_dtype="auto" hoặc float16 để tiết kiệm VRAM cho card 4GB
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32#,            low_cpu_mem_usage=True

        ).to(self.device)
        
        self.model.eval()

    def rerank(self, query: str, chunks: list, top_k: int = 10, batch_size: int = 4):
        """
        query: Câu hỏi
        chunks: List các dict {'text': ...}
        batch_size: Để thấp (4-8) cho card 4GB để tránh OOM
        """
        texts = [ch["text"] for ch in chunks]
        all_scores = []

        # Chia batch thủ công để kiểm soát VRAM
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Encode dữ liệu
            inputs = self.tokenizer(
                [(query, t) for t in batch_texts],
                padding=True,
                truncation=True,#max_length=512,
                
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                # Lấy logits từ model
                outputs = self.model(**inputs)
                # Đa số các reranker dùng logit ở index 0 hoặc single logit
                if outputs.logits.shape[1] == 1:
                    scores = outputs.logits.view(-1).float().cpu().numpy()
                else:
                    # Một số model cũ dùng 2 đầu ra (0: không liên quan, 1: liên quan)
                    scores = outputs.logits[:, 1].float().cpu().numpy()
                
                all_scores.extend(scores)

        # Zip và sắp xếp
        ranked = sorted(
            zip(chunks, all_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]