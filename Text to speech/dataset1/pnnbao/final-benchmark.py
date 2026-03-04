import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from jiwer import cer, wer
import whisper
import unicodedata
import glob

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG'

current_dir = os.path.dirname(os.path.abspath(__file__))
search_pattern = os.path.join(current_dir, "**", "vieneu", "__init__.py")
found_files = glob.glob(search_pattern, recursive=True)

if found_files:
    package_root = os.path.dirname(os.path.dirname(found_files[0]))
    if package_root not in sys.path: sys.path.append(package_root)
else:
    print("❌ Không tìm thấy thư mục 'vieneu'."); sys.exit()

try:
    from vieneu import Vieneu
except ImportError as e:
    print(f"❌ Lỗi Import: {e}"); sys.exit()

BASE_PATH = r"F:\Dataset1\tts-pnnBao"
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights") 

VIVOS_TEST_DIR = os.path.join(BASE_PATH, "vivos_data", "vivos", "test")
PROMPT_PATH = os.path.join(VIVOS_TEST_DIR, "prompts.txt")
SAVE_CSV = os.path.join(BASE_PATH, "ket_qua_pnnbao_final.csv")

def run_benchmark():
    print(f"--- Đang nạp model từ thư mục: {WEIGHTS_DIR} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    try:
        tts_model = Vieneu(model_path=WEIGHTS_DIR, device=device)
    except TypeError:
       
        tts_model = Vieneu(WEIGHTS_DIR)

    print("--- Đang nạp Whisper Small ---")
    asr_model = whisper.load_model("small", device=device)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    print(f"Bắt đầu Benchmark {len(lines)} mẫu...")

    for i, line in enumerate(lines):
        parts = line.strip().split(' ', 1)
        if len(parts) < 2: continue
        text_in = unicodedata.normalize('NFC', parts[1].lower().strip())

        try:
            start_time = time.time()
            audio = tts_model.infer(text=text_in)
            latency = time.time() - start_time

            duration = len(audio) / 24000 # Sample rate mặc định của VieNeu
            rtf = latency / duration if duration > 0 else 0
            
            
            res_asr = asr_model.transcribe(audio.astype(np.float32), language="vi", fp16=(device=="cuda"))
            text_out = unicodedata.normalize('NFC', res_asr["text"].lower().strip())

            results.append({'latency': latency, 'rtf': rtf, 'cer': cer(text_in, text_out), 'wer': wer(text_in, text_out)})

           
            if (i + 1) % 50 == 0:
                cur_cer = np.mean([r['cer'] for r in results])
                print(f"📍 Mẫu {i+1}/{len(lines)} | CER TB hiện tại: {cur_cer:.2%}")

        except Exception as e:
            print(f"⚠️ Lỗi tại mẫu {i+1}: {e}"); continue

    if results:
        df = pd.DataFrame(results)
        summary = pd.DataFrame({
            "Chỉ số": ["Latency (Giây)", "Latency (ms)", "RTF trung bình", "CER (%)", "WER (%)"],
            "Kết quả": [
                round(df['latency'].mean(), 4),
                round(df['latency'].mean() * 1000, 2), # Quy ra ms
                round(df['rtf'].mean(), 4),
                round(df['cer'].mean() * 100, 2),
                round(df['wer'].mean() * 100, 2)
            ]
        })
        print("\n" + "="*45 + "\nKẾT QUẢ CUỐI CÙNG\n" + summary.to_string(index=False) + "\n" + "="*45)
        df.to_csv(SAVE_CSV, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    run_benchmark()