!pip install transformers
from transformers import AutoModel
model = AutoModel.from_pretrained("Scrowbin/whisper-large-v3-vlsp2020_vinai_100h-20000", dtype="auto")

import torch
from transformers import pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="Scrowbin/whisper-large-v3-vlsp2020_vinai_100h-20000",
    chunk_length_s=30,
    device=device,
)

print("✅ Pipeline đã sẵn sàng để nhận dạng!")

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/fleurs_vi

!tar -xf "/content/drive/MyDrive/test.tar.gz" -C /content/fleurs_vi

import pandas as pd
import os

meta_path = "/content/test.tsv"
audio_root = "/content/fleurs_vi/test"

df = pd.read_csv(meta_path, sep="\t")
df.head()

import os

current_audio_col_name = df.columns[1]
current_transcription_col_name = df.columns[2]

df.rename(columns={current_audio_col_name: 'path', current_transcription_col_name: 'transcription'}, inplace=True)

df["audio_path"] = df["path"].apply(lambda x: os.path.join(audio_root, x))
df = df[["audio_path", "transcription"]]
df.head()
preds = []

for i, row in df.iterrows():
    try:
        result = asr_pipe(row["audio_path"])
        preds.append(result["text"])
    except Exception as e:
        print(f"Lỗi file {row['audio_path']}: {e}")
        preds.append("")

    if i % 10 == 0:
        print(f"Đã xử lý {i}/{len(df)} file")
!pip install jiwer
from jiwer import wer, cer

refs = df["transcription"].tolist()
hyps = df["prediction"].tolist()

wer_score = wer(refs, hyps)
print("📊 WER:", wer_score)
cer_score = cer(refs, hyps)
print("📊 CER:", cer_score)
import time
import soundfile as sf
total_audio_duration = 0.0

df_test = df.head(50).copy()

total_audio_duration = 0.0
start_time = time.time()

for i, row in df_test.iterrows():

    with sf.SoundFile(row["audio_path"]) as f:
        total_audio_duration += len(f) / f.samplerate


    result = asr_pipe(row["audio_path"])


total_inference_time = time.time() - start_time
quick_rtf = total_inference_time / total_audio_duration

print(f"🚀 RTF dự kiến cho báo cáo: {quick_rtf}")

