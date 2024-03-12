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

def load_audio(audio_file_path, sample_rate=16000):
    """
    Load the entire audio file and resample if necessary.

    Parameters:
    - audio_file_path (str): The file path of the audio file.
    - sample_rate (int): The desired sample rate. Default is 16000.

    Returns:
    - torch.Tensor: The waveform of the audio file.
    - int: The sample rate of the audio file.
    """
    waveform, sr = torchaudio.load(audio_file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform, sample_rate

def tokenize_audio(segment, processor, sample_rate=16000):
    """
    Tokenize the audio segment.

    Parameters:
    - segment (torch.Tensor): The audio segment.
    - processor: The audio processor for tokenization.
    - sample_rate (int): The sample rate of the audio file. Default is 16000.

    Returns:
    - torch.Tensor: The tokenized audio.
    """
    segment = segment.mean(dim=0)
    tokens = processor(
        segment.numpy(), 
        return_tensors="pt", 
        padding="max_length", 
        max_length=30000, 
        truncation=True, 
        sampling_rate=sample_rate
    ).input_values
    return tokens

def extract_features(tokens, model):
    """
    Extract features using the model.

    Parameters:
    - tokens (torch.Tensor): The tokenized audio.
    - model: The wav2vec2 model for feature extraction.

    Returns:
    - torch.Tensor: The extracted features.
    """
    with torch.no_grad():
        features = model.wav2vec2(tokens).last_hidden_state
        return features

def compute_attention_weights(features, model):
    """
    Compute attention weights.

    Parameters:
    - features (torch.Tensor): The extracted features.
    - model: The model containing the attention mechanism.

    Returns:
    - torch.Tensor: The attention weights.
    """
    attention_weights = model.attention(features)
    return attention_weights

def apply_attention(weights, features):
    """
    Apply attention weights to features.

    Parameters:
    - weights (torch.Tensor): The attention weights.
    - features (torch.Tensor): The extracted features.

    Returns:
    - torch.Tensor: The attended features.
    """
    attended_features = torch.sum(weights * features, dim=1).squeeze().numpy()
    return attended_features

def extract_embeddings(audio_file_path, start, end, processor, model, sample_rate=16000):
    """
    Extract audio embeddings from a specified segment of an audio file.

    Parameters:
    - audio_file_path (str): The file path of the audio file.
    - start (float): The start timestamp (in seconds) of the segment.
    - end (float): The end timestamp (in seconds) of the segment.
    - processor: The audio processor for tokenization.
    - model: The wav2vec2 model for feature extraction.
    - sample_rate (int): The desired sample rate. Default is 16000.

    Returns:
    - numpy.ndarray: The extracted audio embeddings.
    """
    waveform, sr = load_audio(audio_file_path, sample_rate)
    start_sample = int(sr * start)
    end_sample = int(sr * end)
    segment = waveform[:, start_sample:end_sample]
    tokens = tokenize_audio(segment, processor, sample_rate)
    features = extract_features(tokens, model)
    attention_weights = compute_attention_weights(features, model)
    embeddings = apply_attention(attention_weights, features)
    return embeddings


class Wav2Vec2ForClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, config):
        super(Wav2Vec2ForClassification, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            pretrained_model_name, config=config
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Linear(
            self.wav2vec2.config.hidden_size, num_labels
        )
        
    def forward(self, input_values, labels=None):
        """
        Perform forward pass of the model.

        Parameters:
        - input_values: The input audio values.
        - labels: The ground truth labels for training. Default is None.

        Returns:
        - tuple or tensor: If labels are provided, returns a tuple containing the loss and the logits.
                        Otherwise, returns just the logits.
        """
        # Extract features from audio
        features = self.wav2vec2(input_values).last_hidden_state
        
         # Compute attention weights
        attention_weights = self.attention(features)
        
        # Apply attention weights to features
        attended_features = torch.sum(attention_weights * features, dim=1)

        #pooled = features.mean(dim=1)

        # Classification layer
        logits = self.classifier(attended_features)
        
        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        # If labels provided, return loss, otherwise just logits
        return (loss, logits) if loss is not None else logits

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2_meld_pretrained")

# Load the pretrained model configuration
config = Wav2Vec2Config.from_pretrained(
    'facebook/wav2vec2-base-960h', dropout=0.5, attention_dropout=0.5
)

# Initialize the model for classification
model = Wav2Vec2ForClassification(
    "facebook/wav2vec2-base-960h", 7, config=config
)

# Load the pretrained model weights
model.load_state_dict(
    torch.load("./wav2vec2_meld_pretrained/pytorch_model.bin")
)

# Set the model to evaluation mode
model.eval()

# Read the paths of valid embedding folders from a CSV file
valid_paths = pd.read_csv("valid_embedding_folders.csv")

# Loop through each valid folder path
for path in tqdm(valid_paths):
    # Skip if embeddings file already exists
    if os.path.exists(os.path.join(path, 'wav2_vec2_finetuned_embeddings.npz')):
        continue
    
    # Record the start time
    t_start = time.time()
    
    # Load syncmap JSON file
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    
    embeddings_list = []
    
    # Iterate over each video file in the folder
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            audio_path = os.path.join(path, file)
            
            # Extract embeddings for each fragment in the syncmap
            for i, fragment in enumerate(syncmap['fragments']):
                begin = float(fragment["begin"])
                if i == (len(syncmap['fragments'])-1):
                    end = float(fragment["end"])
                else:
                    end = float(syncmap['fragments'][i+1]["begin"])
                
                # Extract embeddings for the audio segment
                embeddings = extract_embeddings(audio_path, begin, end,  processor, model)
                embeddings_list.append(embeddings)
    
    # Concatenate embeddings from all fragments
    embeddings_array = np.vstack(embeddings_list)
    
    # Save the embeddings as a .npz file
    np.savez(os.path.join(path, 'wav2_vec2_finetuned_embeddings.npz'), arr_0=embeddings_array)
    
    # Print the time taken for processing the folder
    print("Time for day: ", time.time()-t_start)