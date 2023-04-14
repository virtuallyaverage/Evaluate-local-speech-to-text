import whisper
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

start_time = time.process_time()
model = whisper.load_model("small.en")
options = dict(beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)

def transcribe_audio_file(audio_file_path):
    result = model.transcribe(audio_file_path)
    transcription = result["text"].strip()
    return transcription
print(f"setup in {time.process_time()-start_time}seconds")
start = time.process_time()

for _ in range(10):

    transcription = transcribe_audio_file('E:\coding\AI\Speech_recognition\Evaluate-local-speech-to-text\say_hi_to_matt.wav')
    
print(f"completed in {10/(time.process_time()-start)}s/it")
print(transcription)