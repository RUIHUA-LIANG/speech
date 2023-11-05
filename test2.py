import whisper
import torch

# Load audio file
audio_data = whisper.load_audio("audio.mp3")

# Load Whisper model
model = whisper.load_model("base")

# Move the model to multiple GPUs
model.encoder.to("cuda:0")
model.decoder.to("cuda:1")

# Split audio into chunks of 30 seconds and transcribe each chunk
results = []
for chunk in whisper.split_audio(audio_data):
    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to("cuda:0")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Append the recognized text to the results list
    results.append(result.text)

# Concatenate the results and print the recognized text
print("".join(results))