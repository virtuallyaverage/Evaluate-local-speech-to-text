import torch
import whisper
from tqdm import tqdm
import os

print("getting data")
# Create a dictionary to store the mapping of audio file names to their transcriptions
transcription_mapping = {}

# Replace this path with the path to the folder containing the transcription text files
transcription_files_folder = '/path/to/transcription/files/folder'

# Iterate through the transcription text files
for root, _, files in os.walk(transcription_files_folder):
    for file in files:
        if file.lower().endswith('.txt'):
            with open(os.path.join(root, file), 'r') as transcript_file:
                for line in transcript_file:
                    audio_name, transcription = line.strip().split(' ', 1)
                    transcription_mapping[audio_name] = transcription

# Build the correct_transcriptions list using the audio_files_list and transcription_mapping
correct_transcriptions = [transcription_mapping[os.path.splitext(os.path.basename(audio_file))[0]] for audio_file in audio_files_list]

correct_transcriptions = []
for audio_name, transcription_path in transcription_mapping.items():
    with open(transcription_path, 'r') as file:
        correct_transcriptions.append(file.readline().strip())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

model = whisper.load_model("small.en")
options = dict(beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)
def transcribe_audio_file(audio_file_path):
    result = model.transcribe(audio_file_path)
    transcription = result["text"].strip()
    return transcription

transcriptions = []

for audio_file in tqdm(audio_files_list):
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    transcription = model.transcribe(mel, **transcribe_options)["text"]
    transcriptions.append(transcription)
