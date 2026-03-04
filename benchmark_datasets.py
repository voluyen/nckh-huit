"""
Benchmark Datasets for Vietnamese Chatbot Evaluation
Sử dụng các dataset công khai từ Hugging Face và GitHub
"""

import streamlit as st
from datasets import load_dataset
import pandas as pd
from typing import List, Dict
import json
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class BenchmarkDatasets:
    """Quản lý các dataset benchmark cho đánh giá chatbot"""
    
    def __init__(self):
        self.datasets = {
            "vietnamese_qa": {
                "name": "ViMMRC 2.0",
                "description": "Vietnamese Multi-domain Machine Reading Comprehension v2.0",
                "source": "uitnlp/vimmrc2.0",
                "type": "QA",
                "size": "~2k câu hỏi"
            },
            "coding": {
                "name": "HumanEval",
                "description": "OpenAI's coding benchmark - Python programming problems",
                "source": "openai_humaneval",
                "type": "Code Generation",
                "size": "164 problems"
            },
            "boolq": {
                "name": "BoolQ",
                "description": "Câu hỏi Yes/No từ Google",
                "source": "google/boolq",
                "type": "Boolean QA",
                "size": "~16k câu hỏi"
            },
            "squad_v2": {
                "name": "SQuAD v2",
                "description": "Reading comprehension dataset",
                "source": "squad_v2",
                "type": "Reading Comprehension",
                "size": "~150k câu hỏi"
            },
            "common_sense": {
                "name": "CommonsenseQA",
                "description": "Câu hỏi về kiến thức thường thức",
                "source": "tau/commonsense_qa",
                "type": "Multiple Choice",
                "size": "~12k câu hỏi"
            }
        }
    
    def load_vietnamese_qa_samples(self, num_samples: int = 10) -> List[Dict]:
        """Tải mẫu câu hỏi tiếng Việt từ ViMMRC 2.0"""
        try:
            # Dataset ViMMRC 2.0 từ Hugging Face
            dataset = load_dataset("uitnlp/vimmrc2.0", split="train")
            samples = []
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # ViMMRC 2.0 có cấu trúc: context, question, answers
                context = item.get('context', '')
                question = item.get('question', '')
                answers = item.get('answers', {})
                answer_text = answers.get('text', [''])[0] if answers else ''
                
                samples.append({
                    "question": question,
                    "context": context[:200] + "..." if len(context) > 200 else context,
                    "answer": answer_text
                })
            
            return samples
        except Exception as e:
            st.warning(f"Không thể tải ViMMRC 2.0: {e}. Sử dụng dataset dự phòng.")
            return self._get_fallback_vietnamese_samples()
    
    def load_coding_samples(self, num_samples: int = 10) -> List[Dict]:
        """Tải mẫu coding problems từ HumanEval"""
        try:
            # Dataset HumanEval từ OpenAI
            dataset = load_dataset("openai_humaneval", split="test")
            samples = []
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                task_id = item.get('task_id', '')
                prompt = item.get('prompt', '')
                entry_point = item.get('entry_point', '')
                
                # Tạo câu hỏi từ prompt
                question = f"[{task_id}] Viết hàm Python: {entry_point}"
                
                samples.append({
                    "question": question,
                    "prompt": prompt[:300] + "..." if len(prompt) > 300 else prompt,
                    "entry_point": entry_point,
                    "task_id": task_id
                })
            
            return samples
        except Exception as e:
            st.warning(f"Không thể tải HumanEval: {e}. Sử dụng dataset dự phòng.")
            return self._get_fallback_coding_samples()
    
    def load_boolq_samples(self, num_samples: int = 10) -> List[Dict]:
        """Tải mẫu BoolQ dataset"""
        try:
            dataset = load_dataset("google/boolq", split="train")
            samples = []
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    "question": item['question'],
                    "passage": item['passage'][:200] + "...",
                    "answer": "Có" if item['answer'] else "Không"
                })
            
            return samples
        except Exception as e:
            st.warning(f"Không thể tải BoolQ: {e}")
            return self._get_fallback_boolq_samples()
    
    def load_squad_samples(self, num_samples: int = 10) -> List[Dict]:
        """Tải mẫu SQuAD v2 dataset"""
        try:
            dataset = load_dataset("squad_v2", split="validation")
            samples = []
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                answer = item['answers']['text'][0] if item['answers']['text'] else "Không có câu trả lời"
                samples.append({
                    "question": item['question'],
                    "context": item['context'][:200] + "...",
                    "answer": answer
                })
            
            return samples
        except Exception as e:
            st.warning(f"Không thể tải SQuAD: {e}")
            return self._get_fallback_squad_samples()
    
    def load_commonsense_samples(self, num_samples: int = 10) -> List[Dict]:
        """Tải mẫu CommonsenseQA dataset"""
        try:
            dataset = load_dataset("tau/commonsense_qa", split="train")
            samples = []
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                choices = item['choices']
                samples.append({
                    "question": item['question'],
                    "choices": list(zip(choices['label'], choices['text'])),
                    "answer": item['answerKey']
                })
            
            return samples
        except Exception as e:
            st.warning(f"Không thể tải CommonsenseQA: {e}")
            return self._get_fallback_commonsense_samples()
    
    def _get_fallback_vietnamese_samples(self) -> List[Dict]:
        """Dataset dự phòng tiếng Việt - ViMMRC 2.0 style"""
        return [
            {
                "question": "Thủ đô của Việt Nam là gì?",
                "context": "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội, một thành phố có lịch sử hơn 1000 năm tuổi.",
                "answer": "Hà Nội"
            },
            {
                "question": "Việt Nam có bao nhiêu tỉnh thành?",
                "context": "Việt Nam được chia thành 63 tỉnh thành, bao gồm 58 tỉnh và 5 thành phố trực thuộc trung ương.",
                "answer": "63 tỉnh thành"
            },
            {
                "question": "Sông dài nhất Việt Nam là gì?",
                "context": "Sông Mê Kông là con sông dài nhất chảy qua Việt Nam, với chiều dài khoảng 4.350 km.",
                "answer": "Sông Mê Kông"
            },
            {
                "question": "Ai là Chủ tịch Hồ Chí Minh?",
                "context": "Chủ tịch Hồ Chí Minh là người sáng lập nước Việt Nam Dân chủ Cộng hòa và là lãnh tụ của cách mạng Việt Nam.",
                "answer": "Người sáng lập nước Việt Nam Dân chủ Cộng hòa"
            },
            {
                "question": "Tết Nguyên Đán là gì?",
                "context": "Tết Nguyên Đán là dịp lễ quan trọng nhất trong năm của người Việt Nam, đánh dấu sự chuyển giao giữa năm cũ và năm mới theo âm lịch.",
                "answer": "Tết cổ truyền của người Việt Nam"
            },
            {
                "question": "Phở là món ăn gì?",
                "context": "Phở là món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt bò hoặc gà.",
                "answer": "Món ăn truyền thống của Việt Nam"
            },
            {
                "question": "Vịnh Hạ Long ở đâu?",
                "context": "Vịnh Hạ Long là một vịnh nổi tiếng thuộc tỉnh Quảng Ninh, được UNESCO công nhận là di sản thiên nhiên thế giới.",
                "answer": "Tỉnh Quảng Ninh"
            },
            {
                "question": "Đồng tiền Việt Nam là gì?",
                "context": "Đồng tiền chính thức của Việt Nam là Việt Nam Đồng, viết tắt là VND.",
                "answer": "Việt Nam Đồng (VND)"
            },
            {
                "question": "Ngôn ngữ chính thức của Việt Nam?",
                "context": "Tiếng Việt là ngôn ngữ chính thức và được sử dụng rộng rãi nhất tại Việt Nam.",
                "answer": "Tiếng Việt"
            },
            {
                "question": "Diện tích Việt Nam là bao nhiêu?",
                "context": "Việt Nam có diện tích khoảng 331,212 km², trải dài từ Bắc vào Nam.",
                "answer": "Khoảng 331,212 km²"
            }
        ]
    
    def _get_fallback_coding_samples(self) -> List[Dict]:
        """Dataset dự phòng coding problems"""
        return [
            {
                "question": "[HumanEval/0] Viết hàm Python: has_close_elements",
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
                "entry_point": "has_close_elements",
                "task_id": "HumanEval/0"
            },
            {
                "question": "[HumanEval/1] Viết hàm Python: separate_paren_groups",
                "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"",
                "entry_point": "separate_paren_groups",
                "task_id": "HumanEval/1"
            },
            {
                "question": "[HumanEval/2] Viết hàm Python: truncate_number",
                "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"",
                "entry_point": "truncate_number",
                "task_id": "HumanEval/2"
            },
            {
                "question": "[HumanEval/3] Viết hàm Python: below_zero",
                "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"",
                "entry_point": "below_zero",
                "task_id": "HumanEval/3"
            },
            {
                "question": "[HumanEval/4] Viết hàm Python: mean_absolute_deviation",
                "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"",
                "entry_point": "mean_absolute_deviation",
                "task_id": "HumanEval/4"
            }
        ]
    
    def _get_fallback_boolq_samples(self) -> List[Dict]:
        """Dataset dự phòng BoolQ"""
        return [
            {"question": "Is Python a programming language?", "answer": "Có"},
            {"question": "Can dogs fly?", "answer": "Không"},
            {"question": "Is the Earth round?", "answer": "Có"},
            {"question": "Is water wet?", "answer": "Có"},
            {"question": "Can humans breathe underwater without equipment?", "answer": "Không"}
        ]
    
    def _get_fallback_squad_samples(self) -> List[Dict]:
        """Dataset dự phòng SQuAD"""
        return [
            {
                "question": "What is AI?",
                "context": "Artificial Intelligence (AI) is intelligence demonstrated by machines...",
                "answer": "Intelligence demonstrated by machines"
            },
            {
                "question": "Who invented the telephone?",
                "context": "Alexander Graham Bell invented the telephone in 1876...",
                "answer": "Alexander Graham Bell"
            }
        ]
    
    def _get_fallback_commonsense_samples(self) -> List[Dict]:
        """Dataset dự phòng CommonsenseQA"""
        return [
            {
                "question": "What do people usually do when they are tired?",
                "choices": [("A", "sleep"), ("B", "run"), ("C", "jump"), ("D", "swim")],
                "answer": "A"
            },
            {
                "question": "Where do you typically find books?",
                "choices": [("A", "library"), ("B", "ocean"), ("C", "sky"), ("D", "car")],
                "answer": "A"
            }
        ]
    
    def get_dataset_info(self) -> pd.DataFrame:
        """Lấy thông tin tất cả datasets"""
        data = []
        for key, info in self.datasets.items():
            data.append({
                "Dataset": info["name"],
                "Mô tả": info["description"],
                "Nguồn": info["source"],
                "Loại": info["type"],
                "Kích thước": info["size"]
            })
        return pd.DataFrame(data)


class ModelBenchmark:
    """Chạy benchmark cho tất cả models"""
    
    def __init__(self, models: List[str]):
        self.models = models
        self.system_prompt = SystemMessage(content="""Bạn là một trợ lý AI thông minh và hữu ích. 
QUAN TRỌNG: Bạn PHẢI LUÔN LUÔN trả lời bằng tiếng Việt, bất kể người dùng hỏi bằng ngôn ngữ gì.
Hãy trả lời một cách tự nhiên, thân thiện và chính xác.""")
    
    def run_single_test(self, model_name: str, question: str, temperature: float = 0.7, timeout: int = 120) -> Dict:
        """Chạy test cho một model với một câu hỏi"""
        try:
            start_time = time.time()
            
            # Khởi tạo model với timeout
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                keep_alive="5m",
                timeout=timeout,  # Add timeout
                num_predict=2000  # Limit response length
            )
            
            # Tạo messages
            messages = [self.system_prompt, HumanMessage(content=question)]
            
            # Gọi model với error handling
            try:
                response = llm.invoke(messages)
                
                # Check if response is valid
                if not response or not hasattr(response, 'content'):
                    raise ValueError("Invalid response from model")
                
                end_time = time.time()
                runtime = end_time - start_time
                
                return {
                    "success": True,
                    "response": response.content,
                    "runtime": runtime,
                    "error": None
                }
            
            except Exception as invoke_error:
                # Model-specific error
                end_time = time.time()
                runtime = end_time - start_time
                error_msg = f"Model invoke error: {str(invoke_error)}"
                
                # Check if it's a timeout
                if runtime >= timeout:
                    error_msg = f"Timeout after {timeout}s"
                
                return {
                    "success": False,
                    "response": None,
                    "runtime": runtime,
                    "error": error_msg
                }
        
        except Exception as e:
            # General error (model not found, connection error, etc.)
            return {
                "success": False,
                "response": None,
                "runtime": 0,
                "error": f"Setup error: {str(e)}"
            }
    
    def run_benchmark(self, questions: List[str], progress_callback=None) -> pd.DataFrame:
        """Chạy benchmark cho tất cả models với danh sách câu hỏi"""
        results = []
        total_tests = len(self.models) * len(questions)
        current_test = 0
        
        for model in self.models:
            model_results = {
                "Model": model,
                "Tổng câu hỏi": len(questions),
                "Thành công": 0,
                "Thất bại": 0,
                "Thời gian TB (s)": 0,
                "Thời gian Min (s)": float('inf'),
                "Thời gian Max (s)": 0,
                "Tổng thời gian (s)": 0
            }
            
            runtimes = []
            
            for i, question in enumerate(questions):
                current_test += 1
                
                if progress_callback:
                    progress_callback(current_test, total_tests, model, i+1, len(questions))
                
                result = self.run_single_test(model, question)
                
                if result["success"]:
                    model_results["Thành công"] += 1
                    runtime = result["runtime"]
                    runtimes.append(runtime)
                    model_results["Tổng thời gian (s)"] += runtime
                    model_results["Thời gian Min (s)"] = min(model_results["Thời gian Min (s)"], runtime)
                    model_results["Thời gian Max (s)"] = max(model_results["Thời gian Max (s)"], runtime)
                else:
                    model_results["Thất bại"] += 1
            
            # Tính trung bình
            if runtimes:
                model_results["Thời gian TB (s)"] = sum(runtimes) / len(runtimes)
            else:
                model_results["Thời gian Min (s)"] = 0
            
            # Làm tròn số
            model_results["Thời gian TB (s)"] = round(model_results["Thời gian TB (s)"], 2)
            model_results["Thời gian Min (s)"] = round(model_results["Thời gian Min (s)"], 2)
            model_results["Thời gian Max (s)"] = round(model_results["Thời gian Max (s)"], 2)
            model_results["Tổng thời gian (s)"] = round(model_results["Tổng thời gian (s)"], 2)
            
            results.append(model_results)
        
        return pd.DataFrame(results)
    
    def run_detailed_benchmark(self, questions: List[str], progress_callback=None) -> tuple:
        """Chạy benchmark chi tiết với từng câu trả lời"""
        summary_results = []
        detailed_results = []
        total_tests = len(self.models) * len(questions)
        current_test = 0
        
        for model in self.models:
            print(f"\n🧪 Testing model: {model}")
            
            model_summary = {
                "Model": model,
                "Tổng câu hỏi": len(questions),
                "Thành công": 0,
                "Thất bại": 0,
                "Thời gian TB (s)": 0,
                "Thời gian Min (s)": float('inf'),
                "Thời gian Max (s)": 0,
                "Tổng thời gian (s)": 0
            }
            
            runtimes = []
            
            for i, question in enumerate(questions):
                current_test += 1
                
                if progress_callback:
                    progress_callback(current_test, total_tests, model, i+1, len(questions))
                
                print(f"  Question {i+1}/{len(questions)}: {question[:50]}...")
                
                result = self.run_single_test(model, question)
                
                # Log result
                if result["success"]:
                    print(f"    ✅ Success in {result['runtime']:.2f}s")
                else:
                    print(f"    ❌ Failed: {result['error']}")
                
                # Lưu kết quả chi tiết
                detailed_results.append({
                    "Model": model,
                    "Câu hỏi": question[:50] + "..." if len(question) > 50 else question,
                    "Câu trả lời": result["response"][:100] + "..." if result["response"] and len(result["response"]) > 100 else result["response"],
                    "Thời gian (s)": round(result["runtime"], 2) if result["success"] else "N/A",
                    "Trạng thái": "✅ Thành công" if result["success"] else f"❌ Lỗi: {result['error']}"
                })
                
                if result["success"]:
                    model_summary["Thành công"] += 1
                    runtime = result["runtime"]
                    runtimes.append(runtime)
                    model_summary["Tổng thời gian (s)"] += runtime
                    model_summary["Thời gian Min (s)"] = min(model_summary["Thời gian Min (s)"], runtime)
                    model_summary["Thời gian Max (s)"] = max(model_summary["Thời gian Max (s)"], runtime)
                else:
                    model_summary["Thất bại"] += 1
                    
                    # Check if we should stop testing this model
                    if "not found" in result["error"].lower() or "connection" in result["error"].lower():
                        print(f"  ⚠️ Skipping remaining questions for {model} due to: {result['error']}")
                        # Mark remaining questions as failed
                        for j in range(i+1, len(questions)):
                            current_test += 1
                            detailed_results.append({
                                "Model": model,
                                "Câu hỏi": questions[j][:50] + "...",
                                "Câu trả lời": None,
                                "Thời gian (s)": "N/A",
                                "Trạng thái": f"❌ Skipped: Model unavailable"
                            })
                            model_summary["Thất bại"] += 1
                        break
            
            # Tính trung bình
            if runtimes:
                model_summary["Thời gian TB (s)"] = sum(runtimes) / len(runtimes)
            else:
                model_summary["Thời gian Min (s)"] = 0
            
            # Làm tròn số
            model_summary["Thời gian TB (s)"] = round(model_summary["Thời gian TB (s)"], 2)
            model_summary["Thời gian Min (s)"] = round(model_summary["Thời gian Min (s)"], 2)
            model_summary["Thời gian Max (s)"] = round(model_summary["Thời gian Max (s)"], 2)
            model_summary["Tổng thời gian (s)"] = round(model_summary["Tổng thời gian (s)"], 2)
            
            summary_results.append(model_summary)
            
            print(f"  📊 {model} completed: {model_summary['Thành công']}/{len(questions)} successful")
        
        return pd.DataFrame(summary_results), pd.DataFrame(detailed_results)


def display_benchmark_ui():
    """Giao diện benchmark trong Streamlit"""
    st.title("📊 Benchmark Datasets")
    st.caption("Đánh giá chatbot với các dataset công khai")
    
    benchmark = BenchmarkDatasets()
    
    # Hiển thị thông tin datasets
    st.subheader("📚 Danh sách Datasets")
    df = benchmark.get_dataset_info()
    st.dataframe(df, use_container_width=True)
    
    # Chọn dataset để test
    st.subheader("🧪 Test với Dataset")
    
    dataset_choice = st.selectbox(
        "Chọn dataset:",
        ["Vietnamese QA", "BoolQ", "SQuAD v2", "CommonsenseQA"]
    )
    
    num_samples = st.slider("Số lượng mẫu:", 5, 20, 10)
    
    if st.button("📥 Tải Dataset", use_container_width=True):
        with st.spinner("Đang tải dataset..."):
            if dataset_choice == "Vietnamese QA":
                samples = benchmark.load_vietnamese_qa_samples(num_samples)
            elif dataset_choice == "BoolQ":
                samples = benchmark.load_boolq_samples(num_samples)
            elif dataset_choice == "SQuAD v2":
                samples = benchmark.load_squad_samples(num_samples)
            else:
                samples = benchmark.load_commonsense_samples(num_samples)
            
            st.session_state.benchmark_samples = samples
            st.success(f"✅ Đã tải {len(samples)} mẫu!")
    
    # Hiển thị samples
    if "benchmark_samples" in st.session_state:
        st.subheader("📝 Mẫu câu hỏi")
        
        for i, sample in enumerate(st.session_state.benchmark_samples[:5]):
            with st.expander(f"Mẫu {i+1}"):
                st.json(sample)


if __name__ == "__main__":
    display_benchmark_ui()
