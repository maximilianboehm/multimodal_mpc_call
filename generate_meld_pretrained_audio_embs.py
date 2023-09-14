import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import moviepy.editor as mp
import torch

# 1. Load the saved model and tokenizer
model_path = "path_to_save_directory"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

model.eval()  # Set model to evaluation mode

# 2. Function to extract subclips from the MP4 file
def extract_audio_from_subclip(video_path, start_time, end_time, target_sample_rate=16000):
    audio = mp.VideoFileClip(video_path).subclip(start_time, end_time).audio
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path, fps=target_sample_rate)
    waveform, _ = torchaudio.load(audio_path)
    return waveform.squeeze()

# 3. Function to generate embeddings
def generate_embedding(waveform):
    with torch.no_grad():
        input_values = tokenizer(waveform, return_tensors="pt").input_values
        outputs = model(input_values).last_hidden_state  # This gives outputs for each time step
        embedding = outputs.mean(dim=1).squeeze().numpy()  # Average over time steps to get a single vector of shape (768,)
    return embedding

# 4. Iterate over your list
video_path = "path_to_your_video.mp4"
timestamps = [
    # Example format: (start_time, end_time)
    (0, 10),  # from 0s to 10s
    (10, 20),  # from 10s to 20s
    # ... add more timestamps
]

embeddings = []

for start, end in timestamps:
    waveform = extract_audio_from_subclip(video_path, start, end)
    embedding = generate_embedding(waveform)
    embeddings.append(embedding)

embeddings = torch.stack(embeddings)
print(embeddings.shape)  # Should print [number_of_subclips, 768]
