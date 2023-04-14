import torch
import whisper
from tqdm import tqdm
import os

print("getting files")
folder_path = "/path/to/your/audio/folder"
audio_files_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.wav', '.mp3', '.flac'))]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

model = whisper.load_model("base")
options = dict(beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)

transcriptions = []

for audio_file in tqdm(audio_files_list):
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    transcription = model.transcribe(mel, **transcribe_options)["text"]
    transcriptions.append(transcription)
