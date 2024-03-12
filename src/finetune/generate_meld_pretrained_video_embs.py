import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-ac3c47b3-456e-56ff-aa3e-5731e429d659"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitImageProcessor, BeitModel, BeitConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time
import sys

def extract_frames(video_file_path, start_frame, end_frame):
    """
    Extract frames from the video file within the specified frame range.

    Parameters:
    - video_file_path (str): The file path of the video file.
    - start_frame (int): The starting frame index.
    - end_frame (int): The ending frame index.

    Returns:
    - List of numpy arrays: List of frames extracted from the video.
    """
    try:
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(start_frame, end_frame, 10):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    except:
        print("Error extracting frames from video.")
        return None

def extract_features(processor, model, frames):
    """
    Extract features from frames using the specified processor and model.

    Parameters:
    - processor: The image processor.
    - model: The BEiT model.
    - frames (list of numpy arrays): List of frames to extract features from.

    Returns:
    - numpy.ndarray: Array of embeddings extracted from frames.
    """
    embeddings_list = []
    for frame in frames:
        inputs = processor(frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model.beit(pixel_values=inputs['pixel_values'])
            features = outputs.last_hidden_state
            attention_weights = model.attention(features)
            embeddings = torch.sum(attention_weights * features, dim=1).squeeze().numpy()
            embeddings_list.append(embeddings)
    return np.vstack(embeddings_list)

def extract_embeddings(video_file_path, start, end, processor, model):
    """
    Extract embeddings from a specified segment of a video file.

    Parameters:
    - video_file_path (str): The file path of the video file.
    - start (float): The start timestamp (in seconds) of the segment.
    - end (float): The end timestamp (in seconds) of the segment.
    - processor: The image processor.
    - model: The BEiT model.

    Returns:
    - numpy.ndarray: Array of embeddings extracted from the video segment.
    """
    fps = int(cv2.VideoCapture(video_file_path).get(cv2.CAP_PROP_FPS))
    start_frame = int(start * fps)
    end_frame = int(end * fps)

    frames = extract_frames(video_file_path, start_frame, end_frame)
    if frames is not None:
        return extract_features(processor, model, frames)
    else:
        return None



class VideoClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, config):
        super(VideoClassification, self).__init__()
        
        #self.beit = BeitForImageClassification.from_pretrained(pretrained_model_name, config=config)
        self.beit = BeitModel.from_pretrained(pretrained_model_name, config=config)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.beit.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Add a classification layer on top
        self.classifier = nn.Linear(self.beit.config.hidden_size, num_labels)
        
    def forward(self, pixel_values, labels=None):
        """
        Forward pass of the BEiT model.

        Parameters:
        - pixel_values (torch.Tensor): Tensor representing the input pixel values of the images.
        - labels (torch.Tensor or None): Tensor representing the ground-truth labels for classification. Default is None.

        Returns:
        - Tuple or torch.Tensor: If labels are provided, returns a tuple containing the loss and logits.
            - loss (torch.Tensor): Computed cross-entropy loss.
            - logits (torch.Tensor): Output logits from the classification layer.
        - If labels are not provided, returns logits only.

        Note:
        - This method extracts features from images using the BEiT model, computes attention weights,
        applies attention weights to features, and performs classification using the classifier layer.
        """
        # Extract features from images
        outputs = self.beit(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        
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
        
        return (loss, logits) if loss is not None else logits

# Load the processor and model
processor = BeitImageProcessor.from_pretrained("./BEiT_meld_pretrained")

# Load the BEiT configuration
config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224-pt22k', dropout=0.4, attention_dropout=0.4)

# Initialize the video classification model using the BEiT architecture
model = VideoClassification("microsoft/beit-base-patch16-224-pt22k", 7, config=config)

# Load the pretrained model weights
model.load_state_dict(torch.load("./BEiT_meld_pretrained/pytorch_model.bin"))

# Set the model to evaluation mode
model.eval()

# Read the paths to valid embedding folders from a CSV file
valid_paths = pd.read_csv("valid_embedding_folders.csv")

# Loop through each valid path
for path in tqdm(valid_paths):
    print()
    print(path)
    skip = False
    
    # Check if the embeddings file already exists
    if os.path.exists(os.path.join(path, 'BEiT_finetuned_embeddings.npz')):
        continue
    
    # Record the starting time for processing the video
    t_start = time.time()
    
    # Open the syncmap JSON file to retrieve synchronization information
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    
    embeddings_list = []
    
    # Loop through each video file in the folder
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            video_path = os.path.join(path, file)
            print()
            print(video_path)
            
            # Loop through each fragment in the syncmap
            for i, fragment in enumerate(syncmap['fragments']):
                begin = float(fragment["begin"])
                if i == (len(syncmap['fragments'])-1):
                    end = float(fragment["end"])
                else:
                    end = float(syncmap['fragments'][i+1]["begin"])
                
                # Extract embeddings for the video segment
                embeddings = extract_embeddings(video_path, begin, end, processor, model)
                
                # Check for data errors during embedding extraction
                if embeddings is None:
                    skip = True
                    break
                
                embeddings_list.append(embeddings)
    
    # Skip processing if any data error occurred
    if skip:
        continue
    
    # Stack the embeddings into an array
    embeddings_array = np.vstack(embeddings_list)
    
    # Save the embeddings array to an NPZ file
    np.savez(os.path.join(path, 'BEiT_finetuned_embeddings.npz'), arr_0=embeddings_array)
    
    # Print the processing time for the video
    print("Time for day video: ", time.time()-t_start)
