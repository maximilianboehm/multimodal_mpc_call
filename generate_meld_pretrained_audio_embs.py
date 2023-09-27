import os
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import json
import time
import pandas as pd
import numpy as np

class Wav2Vec2ForClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, config):
        super(Wav2Vec2ForClassification, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)
        
    def forward(self, input_values, labels=None):
        # Extract features from audio
        features = self.wav2vec2(input_values).last_hidden_state
        
        # Pooling
        # Here mean pooling
        
         # Compute attention weights
        attention_weights = self.attention(features)
        
        # Apply attention weights to features
        attended_features = torch.sum(attention_weights * features, dim=1)

        #pooled = features.mean(dim=1)

        # Classification layer
        logits = self.classifier(attended_features) #pooled
        
        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        # If labels provided, return loss, otherwise just logits
        return (loss, logits) if loss is not None else logits

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2_meld_pretrained")

config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base-960h', dropout=0.5, attention_dropout=0.5)
model = Wav2Vec2ForClassification("facebook/wav2vec2-base-960h", 7, config=config)
model.load_state_dict(torch.load("./wav2vec2_meld_pretrained/pytorch_model.bin"))

# Set the model to evaluation mode
model.eval()

def extract_embeddings(audio_file_path, start, end, processor, model, sample_rate=16000):
    # Load the entire audio file
    waveform, sr = torchaudio.load(audio_file_path)
    
    # If the sample rate is not the desired rate, resample it.
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert start and end timestamps to sample indices
    start_sample = int(sr * start) 
    end_sample = int(sr * end)

    # Extract the desired segment of the waveform
    segment = waveform[:, start_sample:end_sample]
    
    segment = segment.mean(dim=0)
    
    # Tokenize
    tokens = processor(
        segment.numpy(), 
        return_tensors="pt", 
        padding="max_length", 
        max_length=30000, 
        truncation=True, 
        sampling_rate=sample_rate
    ).input_values
    # Extract features using the model
    with torch.no_grad():
        features = model.wav2vec2(tokens).last_hidden_state

        # Compute attention weights
        attention_weights = model.attention(features)
        # Apply attention weights to features
        embeddings = torch.sum(attention_weights * features, dim=1).squeeze().numpy()

    return embeddings

valid_paths = pd.read_csv("valid_embedding_folders.csv")
for path in tqdm(valid_paths):
    if os.path.exists(os.path.join(path, 'wav2_vec2_finetuned_embeddings.npz')):
        continue
    t_start = time.time()
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    embeddings_list = []
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            audio_path = os.path.join(path, file)
            print()
            print(audio_path)
            for i, fragment in enumerate(syncmap['fragments']):
                begin = float(fragment["begin"])
                if i == (len(syncmap['fragments'])-1):
                    end = float(fragment["end"])
                else:
                    end = float(syncmap['fragments'][i+1]["begin"])
                embeddings = extract_embeddings(audio_path, begin, end,  processor, model)
                embeddings_list.append(embeddings)
    embeddings_array = np.vstack(embeddings_list)
    np.savez(os.path.join(path, 'wav2_vec2_finetuned_embeddings.npz'), arr_0=embeddings_array)
    print("Time for day: ", time.time()-t_start)