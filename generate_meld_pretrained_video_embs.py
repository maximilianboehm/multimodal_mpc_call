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

config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224-pt22k', dropout=0.4, attention_dropout=0.4)
model = VideoClassification("microsoft/beit-base-patch16-224-pt22k", 7, config=config)
model.load_state_dict(torch.load("./BEiT_meld_pretrained/pytorch_model.bin"))
model.eval()

def extract_embeddings(video_file_path,start, end, processor, model):
    try:
        cap = cv2.VideoCapture(video_file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start*fps)
        end_frame = int(end*fps)
        frames = []
        for i in range(start_frame, end_frame, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

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
    except:
        print("Data Error in syncmap or video file")
        return None

# Example usage
valid_paths = pd.read_csv("valid_embedding_folders.csv")
for path in tqdm(valid_paths): 
    print()
    print(path)
    skip = False
    if os.path.exists(os.path.join(path, 'BEiT_finetuned_embeddings.npz')):
        continue
    t_start = time.time()
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    embeddings_list = []
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            video_path = os.path.join(path, file)
            print()
            print(video_path)
            for i, fragment in enumerate(syncmap['fragments']):
                begin = float(fragment["begin"])
                if i == (len(syncmap['fragments'])-1):
                    end = float(fragment["end"])
                else:
                    end = float(syncmap['fragments'][i+1]["begin"])
                embeddings = extract_embeddings(video_path, begin, end,  processor, model)
                if embeddings is None:
                    skip = True
                    break
                embeddings_list.append(embeddings)
    if skip:
        continue
    embeddings_array = np.vstack(embeddings_list)
    np.savez(os.path.join(path, 'BEiT_finetuned_embeddings.npz'), arr_0=embeddings_array)
    print("Time for day video: ", time.time()-t_start)