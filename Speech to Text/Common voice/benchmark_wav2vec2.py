import os
import time
import pandas as pd
import librosa
from tqdm import tqdm
from jiwer import wer, cer
from transformers import pipeline

# =============================
# LOAD DATASET
# =============================

BASE_PATH = "datasets/vi"
TSV_PATH = os.path.join(BASE_PATH, "test.tsv")
CLIPS = os.path.join(BASE_PATH, "clips")

df = pd.read_csv(TSV_PATH, sep="\t")
df = df.head(1000)

print("Loaded samples:", len(df))

# =============================
# LOAD MODEL
# =============================

print("\nLoading Wav2Vec2 Vietnamese...")
asr = pipeline(
    "automatic-speech-recognition",
    model="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    device=-1
)
print("Model ready!\n")

# =============================
# BENCHMARK
# =============================

references = []
predictions = []

total_audio_time = 0
total_infer_time = 0

for i, row in tqdm(df.iterrows(), total=len(df)):

    audio_path = os.path.join(CLIPS, row["path"])
    text_ref = str(row["sentence"]).lower()

    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        duration = len(audio) / 16000
        total_audio_time += duration

        start = time.time()
        result = asr(audio)
        end = time.time()

        pred = result["text"].lower()

        total_infer_time += (end - start)

        references.append(text_ref)
        predictions.append(pred)

    except Exception as e:
        print("Skip:", audio_path, e)

# =============================
# METRICS
# =============================

final_wer = wer(references, predictions)
final_cer = cer(references, predictions)
rtf = total_infer_time / total_audio_time

print("\n===== RESULT =====")
print("WER:", final_wer)
print("CER:", final_cer)
print("RTF:", rtf)