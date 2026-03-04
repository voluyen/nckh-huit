import os
import time
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from jiwer import wer, cer
import whisper
from tqdm import tqdm

BASE_DIR = r"F:\Dataset3\metamms_model"
# Tận dụng dataset FLEURS từ thư mục có sẵn của bạn
DATASET_DIR = r"F:\Dataset3\pnnBaomodel\pnnBaomodel\fleurs_data"
METADATA_FILE = os.path.join(DATASET_DIR, "test.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "mms_benchmark_results.csv")

os.environ['HF_HOME'] = os.path.join(BASE_DIR, "hf_cache")

def main():
    print("-> Đang tải Meta MMS-TTS (vietnamese)...")
    model_id = "facebook/mms-tts-vie"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    
    print("-> Đang tải Whisper (base) làm giám khảo...")
    asr_model = whisper.load_model("base")

    if not os.path.exists(METADATA_FILE):
        print(f"❌ Không tìm thấy file dữ liệu tại: {METADATA_FILE}")
        return
        
    df = pd.read_csv(METADATA_FILE, sep='\t', header=None)
    results = []
    current_batch_cers = []

    print(f"-> Bắt đầu Benchmark {len(df)} mẫu dataset FLEURS...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        raw_text = str(row[3]).strip()
        
        try:
            inputs = tokenizer(raw_text, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                output = model(**inputs).waveform
            infer_duration = time.time() - start_time
            
            audio_data = output.cpu().numpy().squeeze()
            sr = model.config.sampling_rate # 16000Hz cho MMS
            audio_len_sec = len(audio_data) / sr
            
            current_rtf = infer_duration / audio_len_sec
            
            temp_wav = "temp_mms_eval.wav"
            sf.write(temp_wav, audio_data, sr)
            
            asr_res = asr_model.transcribe(temp_wav, language="vi", fp16=False)
            pred_text = asr_res["text"].strip().lower()
            ref_text = raw_text.lower()
            
            c_wer = wer(ref_text, pred_text)
            c_cer = cer(ref_text, pred_text)
            
            results.append({
                "latency_ms": infer_duration * 1000,
                "rtf": current_rtf,
                "wer": c_wer,
                "cer": c_cer
            })
            current_batch_cers.append(c_cer)
            
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                # In thông báo nhanh CER batch hiện tại
                avg_batch_cer = np.mean(current_batch_cers) * 100
                print(f"\n[Thông báo] Đã xong {i+1} mẫu. CER batch này: {avg_batch_cer:.2f}%")
                current_batch_cers = []

        except Exception as e:
            print(f"⚠️ Bỏ qua mẫu tại dòng {i} do lỗi: {e}")
            continue

    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*50)
    print(" BÁO CÁO TỔNG KẾT HIỆU NĂNG META MMS")
    print("-" * 50)
    print(f"1. Latency trung bình: {final_df['latency_ms'].mean():.2f} ms")
    print(f"2. RTF trung bình:     {final_df['rtf'].mean():.4f}")
    print(f"3. WER trung bình:     {final_df['wer'].mean()*100:.2f} %")
    print(f"4. CER trung bình:     {final_df['cer'].mean()*100:.2f} %")
    print("-" * 50)
    print(f" Tổng số mẫu thành công: {len(final_df)}")
    print(f" Kết quả chi tiết lưu tại: {OUTPUT_CSV}")
    print("="*50)

if __name__ == "__main__":
    main()