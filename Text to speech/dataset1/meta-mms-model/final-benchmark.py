import os
import time
import torch
import numpy as np
import pandas as pd
from jiwer import cer, wer
import whisper
import warnings
import logging
from transformers import VitsModel, VitsTokenizer
import unicodedata

warnings.filterwarnings("ignore")
logging.getLogger('numba').setLevel(logging.WARNING)

DATA_DIR = r"F:\Dataset1\tts-thanhtantran\vivos_data\vivos\test"
PROMPT_PATH = os.path.join(DATA_DIR, "prompts.txt")
SAVE_CSV = r"F:\Dataset1\meta_tts\ket_qua_meta_vivos.csv"
MODEL_ID = "facebook/mms-tts-vie"

def run_benchmark():
    if not os.path.exists(os.path.dirname(SAVE_CSV)):
        os.makedirs(os.path.dirname(SAVE_CSV))

    print("--- Dang nap Whisper & Meta MMS ---")
    asr_model = whisper.load_model("small")
    tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    print(f"Bat dau benchmark {len(lines)} mau...")

    for i, line in enumerate(lines):
        parts = line.strip().split(' ', 1)
        if len(parts) < 2: continue
        text_in = parts[1].lower().strip()

        try:
            text_in = unicodedata.normalize('NFC', text_in)
            inputs = tokenizer(text_in, return_tensors="pt").to(device)
            
            t0 = time.time()
            with torch.no_grad():
                output = model(**inputs).waveform.cpu().numpy().squeeze()
            t1 = time.time()

            latency = t1 - t0
            duration = len(output) / 16000 
            rtf = latency / duration if duration > 0 else 0

            res_asr = asr_model.transcribe(output.astype(np.float32), language="vi", fp16=False)
            text_out = res_asr["text"].lower().strip()

            results.append({
                'latency': latency,
                'rtf': rtf,
                'cer': cer(text_in, text_out),
                'wer': wer(text_in, text_out)
            })

            if (i + 1) % 50 == 0:
                print(f"-> Xong {i+1}/{len(lines)}. CER: {results[-1]['cer']:.2%} | RTF: {rtf:.4f}")
        
        except Exception:
            continue

    if results:
        df = pd.DataFrame(results)
        summary = pd.DataFrame({
            "Metric": ["Latency (s)", "RTF", "CER (%)", "WER (%)"],
            "Trung binh": [
                round(df['latency'].mean(), 4),
                round(df['rtf'].mean(), 4),
                round(df['cer'].mean() * 100, 2),
                round(df['wer'].mean() * 100, 2)
            ]
        })
        summary.to_csv(SAVE_CSV, index=False, encoding='utf-8-sig')
        print("\n" + "="*30 + "\nKET QUA META MMS\n" + summary.to_string(index=False) + "\n" + "="*30)

if __name__ == "__main__":
    run_benchmark()