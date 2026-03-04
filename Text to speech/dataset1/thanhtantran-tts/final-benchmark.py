import os
import sys
import time
import numpy as np
import pandas as pd
from jiwer import cer, wer
import whisper 

sys.path.insert(0, r"F:\Dataset1\tts-thanhtantran\VieNeu-TTS")

try:
    from vieneu_tts.vieneu_tts import VieNeuTTS
    print("--- Kết nối mô hình VieNeu-TTS thành công ---")
except Exception as e:
    print(f"Lỗi khởi tạo: {e}")
    sys.exit()

DATA_DIR = r"F:\Dataset1\tts-thanhtantran\vivos_data\vivos\test"
PROMPT_PATH = os.path.join(DATA_DIR, "prompts.txt")
SAVE_CSV = r"F:\Dataset1\tts-thanhtantran\ket_qua_vivos_goc.csv"

def find_first_wav(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                return os.path.join(root, file)
    return None

def run_benchmark():
    model = VieNeuTTS()
    asr = whisper.load_model("small")
    
    wav_root = os.path.join(DATA_DIR, "waves")
    ref_wav = find_first_wav(wav_root)
    
    if not ref_wav:
        print(f"Lỗi: Không tìm thấy file wav tại {wav_root}")
        return

    print(f"Sử dụng giọng mẫu: {ref_wav}")
    ref_codes = model.encode_reference(ref_wav)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    print(f"Bắt đầu benchmark {len(lines)} mẫu...")

    for i, line in enumerate(lines):
        parts = line.strip().split(' ', 1)
        if len(parts) < 2: continue
        text_in = parts[1].lower().strip()

        t0 = time.time()
        audio = model.infer(text=text_in, ref_codes=ref_codes, ref_text="mẫu")
        t1 = time.time()

        audio_f = audio.astype(np.float32)
        if np.max(np.abs(audio_f)) > 0: audio_f /= np.max(np.abs(audio_f))
        
        res_asr = asr.transcribe(audio_f, language="vi", fp16=False)
        text_out = res_asr["text"].lower().strip()

        # Tính thêm WER
        results.append({
            'rtf': (t1 - t0) / (len(audio) / 24000),
            'latency': t1 - t0,
            'cer': cer(text_in, text_out),
            'wer': wer(text_in, text_out)
        })

        if (i + 1) % 5 == 0:
            print(f"Tiến độ: {i+1}/{len(lines)} | CER: {results[-1]['cer']:.2%} | WER: {results[-1]['wer']:.2%}")

    df = pd.DataFrame(results)
    summary = pd.DataFrame({
        "Metric": ["RTF", "Latency (ms)", "CER (%)", "WER (%)"],
        "Trung bình": [
            round(df['rtf'].mean(), 4),
            round(df['latency'].mean() * 1000, 2),
            round(df['cer'].mean() * 100, 2),
            round(df['wer'].mean() * 100, 2)
        ]
    })
    summary.to_csv(SAVE_CSV, index=False, encoding='utf-8-sig')
    print("\n--- HOÀN THÀNH ---\n", summary)

if __name__ == "__main__":
    run_benchmark()