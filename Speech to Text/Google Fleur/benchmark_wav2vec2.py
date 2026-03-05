!pip install -q transformers datasets jiwer librosa soundfile

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
model.eval()
from google.colab import drive
drive.mount("/content/drive")

!mkdir -p /content/fleurs_vi
!tar -xf "/content/drive/MyDrive/test.tar.gz" -C /content/fleurs_vi

import pandas as pd, os

meta_path = "/content/test.tsv"
audio_root = "/content/fleurs_vi/test"

df = pd.read_csv(meta_path, sep="\t")

df.rename(columns={
    df.columns[1]: "path",
    df.columns[2]: "sentence"
}, inplace=True)

df["audio_path"] = df["path"].apply(lambda x: os.path.join(audio_root, x))
df = df[["audio_path", "sentence"]]

import librosa, time
from jiwer import wer, cer

def transcribe_w2v(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0]

refs, hyps = [], []
total_time, total_dur = 0, 0

for i, row in df.head(50).iterrows():
    audio, sr = librosa.load(row["audio_path"], sr=16000)
    total_dur += len(audio)/sr

    start = time.time()
    pred = transcribe_w2v(audio)
    total_time += time.time() - start

    refs.append(row["sentence"].lower())
    hyps.append(pred.lower())

print("WER:", wer(refs, hyps))
print("CER:", cer(refs, hyps))
print("RTF:", total_time / total_dur)
