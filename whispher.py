from faster_whisper import WhisperModel
import librosa

# Load Whisper model (CPU optimized)
model = WhisperModel("base", compute_type="int8")  
# If GPU available:
# model = WhisperModel("medium", device="cuda")

# Load audio
audio_path = "D:/Final Year Project/Dataset/Real/real056.wav"

# Transcribe
segments, info = model.transcribe(audio_path, language="si")

# Combine segments
transcription = ""
for segment in segments:
    transcription += segment.text + " "

print("Transcription:")
print(transcription.strip())
