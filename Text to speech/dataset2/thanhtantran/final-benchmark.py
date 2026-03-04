import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from librosa import load
from neucodec import NeuCodec
from jiwer import wer, cer
import whisper
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from vieneu_tts.vieneu_tts import VieNeuTTS as Vieneu
    print("-> Kết nối thành công với VieNeuTTS")
except ImportError:
    sys.exit("❌ Không tìm thấy mã nguồn Vieneu. Hãy chạy script trong thư mục VieNeu-Thanhtan")

BASE_DIR = r"F:\Dataset3\thanhtantranmodel"
MODEL_DIR = os.path.join(BASE_DIR, "weights")
DATASET_DIR = r"F:\Dataset3\pnnBaomodel\pnnBaomodel\fleurs_data"
REF_AUDIO_PATH = os.path.join(DATASET_DIR, "test", "78243143814219998.wav") 
METADATA_FILE = os.path.join(DATASET_DIR, "test.tsv") 
OUTPUT_CSV = os.path.join(BASE_DIR, "thanhtan_final_benchmark.csv")

os.environ['HF_HOME'] = os.path.join(BASE_DIR, "hf_cache")

def main():
    print("-> Đang khởi tạo NeuCodec để trích xuất giọng mẫu...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")

    wav, _ = load(REF_AUDIO_PATH, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        fixed_ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    print("✅ Đã trích xuất xong mã giọng mẫu.")

    os.chdir(MODEL_DIR)
    print("-> Đang khởi tạo VieNeu-TTS...")
    tts = Vieneu()
    os.chdir(CURRENT_DIR)

    print("-> Đang tải Whisper (base)...")
    asr_model = whisper.load_model("base")

    df = pd.read_csv(METADATA_FILE, sep='\t', header=None)
    results = []
    
    print(f"-> Bắt đầu Benchmark {len(df)} mẫu (Tốc độ tối ưu)...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        target_text = str(row[3]).strip().lower()
        
        try:
            start_time = time.time()
            
            # Gọi infer với ref_codes đã chuẩn bị sẵn
            # Model sẽ "nhái" theo giọng của file REF_AUDIO_PATH
            audio_data = tts.infer(
                ref_codes=fixed_ref_codes,
                ref_text="văn bản mẫu", 
                text=target_text
            )
            
            infer_duration = time.time() - start_time
            sr = 24000
            
            temp_wav = "temp_thanhtan_final.wav"
            sf.write(temp_wav, audio_data, sr)
            asr_res = asr_model.transcribe(temp_wav, language="vi", fp16=False)
            pred_text = asr_res["text"].strip().lower()
            
            results.append({
                "latency_ms": infer_duration * 1000,
                "rtf": infer_duration / (len(audio_data) / sr),
                "wer": wer(target_text, pred_text),
                "cer": cer(target_text, pred_text)
            })

            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"\n[OK] Xong {i+1} mẫu. RTF hiện tại: {results[-1]['rtf']:.4f}")

        except Exception as e:
            print(f"⚠️ Lỗi mẫu {i+1}: {e}")
            continue

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ HOÀN THÀNH! Kết quả lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()