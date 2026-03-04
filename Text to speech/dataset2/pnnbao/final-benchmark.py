import os
import time
import pandas as pd
import numpy as np
import soundfile as sf
from vieneu import Vieneu
from jiwer import wer, cer
import whisper
from tqdm import tqdm

BASE_DIR = r"F:\Dataset3\pnnBaomodel\pnnBaomodel"
MODEL_DIR = os.path.join(BASE_DIR, "weights")
METADATA_FILE = os.path.join(MODEL_DIR, "test.tsv") 
OUTPUT_CSV = os.path.join(BASE_DIR, "full_benchmark_results.csv")

os.environ['HF_HOME'] = os.path.join(BASE_DIR, "hf_cache")

def main():
    original_path = os.getcwd()
    os.chdir(MODEL_DIR)
    print("-> Đang khởi tạo VieNeu-TTS...")
    try:
        tts = Vieneu() 
    finally:
        os.chdir(original_path)

    print("-> Đang tải Whisper (base)...")
    asr_model = whisper.load_model("base")

    df = pd.read_csv(METADATA_FILE, sep='\t', header=None)
    total_samples = len(df)
    
    results = []
    current_batch_cers = []
    
    print(f"-> Bắt đầu Benchmark TOÀN BỘ {total_samples} mẫu...")

    for i, row in tqdm(df.iterrows(), total=total_samples):
        ref_text = str(row[3]).strip().lower()
        
        start_time = time.time()
        audio_data = tts.infer(text=ref_text)
        end_time = time.time()
        
        infer_duration = end_time - start_time
        audio_len_sec = len(audio_data) / 24000 
        
        temp_wav = "temp_bench.wav"
        sf.write(temp_wav, audio_data, 24000)
        
        asr_res = asr_model.transcribe(temp_wav, language="vi", fp16=False)
        pred_text = asr_res["text"].strip().lower()
        
        current_wer = wer(ref_text, pred_text)
        current_cer = cer(ref_text, pred_text)
        
        results.append({
            "id": i + 1,
            "latency_ms": infer_duration * 1000,
            "rtf": infer_duration / audio_len_sec,
            "wer": current_wer,
            "cer": current_cer
        })
        
        current_batch_cers.append(current_cer)
        
        if (i + 1) % 50 == 0:
            avg_batch_cer = np.mean(current_batch_cers) * 100
            print(f"\n[Thông báo] Đã chạy thành công {i + 1} sample.")
            print(f"[Stats] CER trung bình của 50 mẫu vừa qua: {avg_batch_cer:.2f}%")
            
            
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            current_batch_cers = [] # Reset batch CER

    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*45)
    print(f" HOÀN THÀNH BENCHMARK {total_samples} MẪU")
    print("-" * 45)
    print(f"- Latency TB: {final_df['latency_ms'].mean():.2f} ms")
    print(f"- RTF TB:     {final_df['rtf'].mean():.4f}")
    print(f"- WER TB:     {final_df['wer'].mean()*100:.2f} %")
    print(f"- CER TB:     {final_df['cer'].mean()*100:.2f} %")
    print(f"- Chi tiết tại: {OUTPUT_CSV}")
    print("="*45)

if __name__ == "__main__":
    main()