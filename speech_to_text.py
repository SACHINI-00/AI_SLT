import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load model
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-sinhala")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-sinhala")

# Load audio file
audio, sr = librosa.load("D:/Final Year Project/Dataset/Real/real056.wav", sr=16000)

# Tokenize
input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

# Get logits
with torch.no_grad():
    logits = model(input_values).logits

# Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print(transcription)
