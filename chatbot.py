import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time
from benchmark_datasets import BenchmarkDatasets, ModelBenchmark

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Local Chatbot (Offline)", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Chatbot Local - Qwen 2.5")
st.caption("Chạy offline trên máy của bạn, không cần Internet.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    
    # Tab cho cấu hình và benchmark
    tab1, tab2 = st.tabs(["🤖 Model", "📊 Benchmark"])
    
    with tab1:
        # Danh sách model local với mô tả
        model_options = {
            "qwen2.5": "Qwen 2.5 - Cân bằng tốc độ và chất lượng",
            "llama3.1": "Llama 3.1 - Hiệu suất cao",
            "gemma2:2b": "Gemma 2B - Nhanh và nhẹ",
            "nxphi47/seallm-7b-v2:q4_0": "SeaLLM v2 - Optimized for Southeast Asian languages"
        }
        
        selected_model = st.selectbox(
            "Chọn Model:", 
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # Cấu hình nâng cao
        with st.expander("🔧 Cấu hình nâng cao"):
            temperature = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max tokens", 100, 4000, 2000, 100)
            keep_alive = st.selectbox("Keep alive", ["1m", "5m", "10m", "30m"], index=1)
        
        st.info(f"🎯 Đang sử dụng: {model_options[selected_model]}")
        
        # Thống kê
        if "chat_history" in st.session_state:
            msg_count = len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)])
            st.metric("Số câu hỏi", msg_count)
        
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with tab2:
        st.markdown("### 📊 Test Benchmark")
        
        benchmark = BenchmarkDatasets()
        
        # Chọn chế độ benchmark
        benchmark_mode = st.radio(
            "Chế độ:",
            ["📥 Xem mẫu", "🚀 Chạy Benchmark tự động"],
            key="bench_mode"
        )
        
        if benchmark_mode == "📥 Xem mẫu":
            dataset_choice = st.selectbox(
                "Chọn dataset:",
                ["Vietnamese QA (ViMMRC 2.0)", "HumanEval (Coding)", "BoolQ", "SQuAD v2", "CommonsenseQA"],
                key="bench_dataset"
            )
            
            num_samples = st.slider("Số mẫu:", 3, 10, 5, key="bench_samples")
            
            if st.button("📥 Tải mẫu", use_container_width=True):
                with st.spinner("Đang tải..."):
                    if dataset_choice == "Vietnamese QA (ViMMRC 2.0)":
                        samples = benchmark.load_vietnamese_qa_samples(num_samples)
                    elif dataset_choice == "HumanEval (Coding)":
                        samples = benchmark.load_coding_samples(num_samples)
                    elif dataset_choice == "BoolQ":
                        samples = benchmark.load_boolq_samples(num_samples)
                    elif dataset_choice == "SQuAD v2":
                        samples = benchmark.load_squad_samples(num_samples)
                    else:
                        samples = benchmark.load_commonsense_samples(num_samples)
                    
                    st.session_state.benchmark_samples = samples
                    st.success(f"✅ Đã tải {len(samples)} mẫu!")
        
        else:  # Chạy benchmark tự động
            st.markdown("**Chạy test tự động cho tất cả models**")
            
            # Chọn dataset
            auto_dataset = st.selectbox(
                "Dataset:",
                ["Vietnamese QA (ViMMRC 2.0)", "HumanEval (Coding)", "BoolQ", "SQuAD v2", "CommonsenseQA"],
                key="auto_dataset"
            )
            
            # Số câu hỏi
            num_questions = st.slider("Số câu hỏi:", 3, 10, 5, key="num_questions")
            
            # Hiển thị models sẽ test
            st.info(f"🤖 Sẽ test {len(model_options)} models: {', '.join(model_options.keys())}")
            
            if st.button("🚀 Chạy Benchmark", use_container_width=True, type="primary"):
                # Load questions
                with st.spinner("Đang tải câu hỏi..."):
                    if auto_dataset == "Vietnamese QA (ViMMRC 2.0)":
                        samples = benchmark.load_vietnamese_qa_samples(num_questions)
                    elif auto_dataset == "HumanEval (Coding)":
                        samples = benchmark.load_coding_samples(num_questions)
                    elif auto_dataset == "BoolQ":
                        samples = benchmark.load_boolq_samples(num_questions)
                    elif auto_dataset == "SQuAD v2":
                        samples = benchmark.load_squad_samples(num_questions)
                    else:
                        samples = benchmark.load_commonsense_samples(num_questions)
                    
                    # Extract questions
                    questions = [s.get("question", str(s)) for s in samples]
                
                # Run benchmark
                st.session_state.running_benchmark = True
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, model, q_num, q_total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Testing {model} - Câu hỏi {q_num}/{q_total} ({int(progress*100)}%)")
                
                model_benchmark = ModelBenchmark(list(model_options.keys()))
                summary_df, detailed_df = model_benchmark.run_detailed_benchmark(
                    questions, 
                    progress_callback=update_progress
                )
                
                st.session_state.benchmark_summary = summary_df
                st.session_state.benchmark_detailed = detailed_df
                st.session_state.running_benchmark = False
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Benchmark hoàn thành!")
                st.rerun()
        
        if st.button("ℹ️ Xem thông tin datasets", use_container_width=True):
            st.session_state.show_benchmark_info = True

# --- KHỞI TẠO SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm" not in st.session_state:
    st.session_state.llm = None

if "current_model" not in st.session_state:
    st.session_state.current_model = None

if "current_temperature" not in st.session_state:
    st.session_state.current_temperature = None

if "current_keep_alive" not in st.session_state:
    st.session_state.current_keep_alive = None

# System prompt để bắt buộc trả lời bằng tiếng Việt
SYSTEM_PROMPT = SystemMessage(content="""Bạn là một trợ lý AI thông minh và hữu ích. 
QUAN TRỌNG: Bạn PHẢI LUÔN LUÔN trả lời bằng tiếng Việt, bất kể người dùng hỏi bằng ngôn ngữ gì.
Hãy trả lời một cách tự nhiên, thân thiện và chính xác.""")

# --- KHỞI TẠO MODEL (CHỈ KHI CẦN THIẾT) ---
@st.cache_resource
def get_llm(model_name, _temperature, _keep_alive, _max_tokens):
    """Cache model để tránh khởi tạo lại mỗi lần"""
    return ChatOllama(
        model=model_name,
        temperature=_temperature,
        keep_alive=_keep_alive,
        num_predict=_max_tokens
    )

# Kiểm tra xem có cần khởi tạo lại model không
if (st.session_state.current_model != selected_model or 
    st.session_state.current_temperature != temperature or
    st.session_state.current_keep_alive != keep_alive or
    st.session_state.llm is None):
    with st.spinner(f"🔄 Đang tải model {selected_model}..."):
        try:
            st.session_state.llm = get_llm(selected_model, temperature, keep_alive, max_tokens)
            st.session_state.current_model = selected_model
            st.session_state.current_temperature = temperature
            st.session_state.current_keep_alive = keep_alive
            st.success(f"✅ Model {selected_model} đã sẵn sàng!", icon="✅")
            time.sleep(1)  # Show success message briefly
        except Exception as e:
            st.error(f"❌ Lỗi tải model: {e}")
            st.warning(f"""
            **Kiểm tra:**
            1. Ollama đã chạy chưa? `ollama list`
            2. Model đã cài chưa? `ollama pull {selected_model.split(':')[0]}`
            3. Thử model khác trong dropdown
            """)
            st.stop()

# --- HIỂN thị LỊCH SỬ CHAT ---
# Hiển thị kết quả benchmark nếu có
if "benchmark_summary" in st.session_state and st.session_state.benchmark_summary is not None:
    st.success("🎉 Kết quả Benchmark")
    
    # Bảng tổng hợp
    st.subheader("📊 Bảng so sánh Models")
    
    # Highlight model nhanh nhất
    summary_df = st.session_state.benchmark_summary.copy()
    
    # Tạo styled dataframe
    def highlight_best(s):
        if s.name == "Thời gian TB (s)":
            min_val = s[s > 0].min() if any(s > 0) else 0
            return ['background-color: lightgreen' if v == min_val and v > 0 else '' for v in s]
        elif s.name == "Thành công":
            max_val = s.max()
            return ['background-color: lightgreen' if v == max_val else '' for v in s]
        return ['' for _ in s]
    
    styled_df = summary_df.style.apply(highlight_best)
    st.dataframe(styled_df, use_container_width=True)
    
    # Thống kê nhanh
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fastest_model = summary_df.loc[summary_df["Thời gian TB (s)"].idxmin(), "Model"]
        fastest_time = summary_df["Thời gian TB (s)"].min()
        st.metric("🏆 Model nhanh nhất", fastest_model, f"{fastest_time}s")
    
    with col2:
        most_reliable = summary_df.loc[summary_df["Thành công"].idxmax(), "Model"]
        success_rate = summary_df["Thành công"].max()
        st.metric("✅ Độ tin cậy cao nhất", most_reliable, f"{success_rate} câu")
    
    with col3:
        total_time = summary_df["Tổng thời gian (s)"].sum()
        st.metric("⏱️ Tổng thời gian", f"{total_time:.2f}s")
    
    # Bảng chi tiết
    if "benchmark_detailed" in st.session_state:
        with st.expander("🔍 Xem chi tiết từng câu trả lời"):
            st.dataframe(st.session_state.benchmark_detailed, use_container_width=True)
    
    # Nút xóa kết quả
    if st.button("🗑️ Xóa kết quả benchmark"):
        del st.session_state.benchmark_summary
        del st.session_state.benchmark_detailed
        st.rerun()
    
    st.divider()

# Hiển thị thông tin benchmark nếu được yêu cầu
if st.session_state.get("show_benchmark_info", False):
    with st.expander("📊 Thông tin chi tiết Benchmark Datasets", expanded=True):
        benchmark = BenchmarkDatasets()
        df = benchmark.get_dataset_info()
        st.dataframe(df, use_container_width=True)
        if st.button("Đóng"):
            st.session_state.show_benchmark_info = False
            st.rerun()

# Hiển thị benchmark samples nếu có
if "benchmark_samples" in st.session_state and st.session_state.benchmark_samples:
    st.info("📝 Benchmark samples đã tải. Bạn có thể copy câu hỏi bên dưới để test!")
    with st.expander("🔍 Xem mẫu câu hỏi", expanded=False):
        for i, sample in enumerate(st.session_state.benchmark_samples[:5]):
            st.markdown(f"**Mẫu {i+1}:**")
            if "question" in sample:
                st.code(sample["question"], language="text")
            st.json(sample)
            st.divider()

chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

# --- XỬ LÝ NHẬP LIỆU ---
if user_query := st.chat_input("💬 Nhập câu hỏi của bạn..."):
    
    # Thêm câu hỏi vào lịch sử
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Hiển thị câu hỏi người dùng
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # Xử lý và hiển thị câu trả lời
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Hiển thị loading
            with st.spinner("🤔 Đang suy nghĩ..."):
                start_time = time.time()
                
                # Tối ưu context: chỉ lấy 10 tin nhắn gần nhất để tránh quá tải
                recent_history = st.session_state.chat_history[-10:]
                
                # Thêm system prompt vào đầu để bắt buộc trả lời tiếng Việt
                messages_with_system = [SYSTEM_PROMPT] + recent_history
                
                # Stream response
                stream = st.session_state.llm.stream(messages_with_system)
                
                for chunk in stream:
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        # Cập nhật với cursor typing effect
                        response_placeholder.markdown(full_response + "▌")
                
                # Hiển thị kết quả cuối cùng
                response_placeholder.markdown(full_response)
                
                # Thống kê thời gian phản hồi
                response_time = time.time() - start_time
                st.caption(f"⏱️ Thời gian phản hồi: {response_time:.2f}s")
            
            # Lưu vào lịch sử
            st.session_state.chat_history.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"❌ Lỗi kết nối Ollama: {e}")
            st.warning("""
            **Hướng dẫn khắc phục:**
            1. Đảm bảo Ollama đã được cài đặt
            2. Chạy lệnh: `ollama run {selected_model}`
            3. Kiểm tra model có sẵn: `ollama list`
            """)
            
            # Xóa tin nhắn lỗi khỏi lịch sử
            if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
                st.session_state.chat_history.pop()

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
        🚀 Powered by Ollama & Streamlit | 
        💡 Tip: Sử dụng câu hỏi ngắn gọn để có phản hồi nhanh hơn
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)