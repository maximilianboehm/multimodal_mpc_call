import os

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-ac3c47b3-456e-56ff-aa3e-5731e429d659"

import pandas as pd
import cv2
import torch
from transformers import BeitImageProcessor, BeitForImageClassification, BeitConfig, BeitModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, random_split
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MeldVideoDataset(Dataset):
    def __init__(self, csv_path, video_folder, processor, frames_to_sample=10):
        self.dataframe = pd.read_csv(csv_path)
        self.video_folder = video_folder
        self.processor = processor
        self.frames_to_sample = frames_to_sample
        self.emotion_mapping = {emotion: i for i, emotion in enumerate(self.dataframe["Emotion"].unique())}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        dialogue_id = self.dataframe.iloc[idx]["Dialogue_ID"]
        utterance_id = self.dataframe.iloc[idx]["Utterance_ID"]
        emotion = self.dataframe.iloc[idx]["Emotion"]
        video_file_path = os.path.join(self.video_folder, f"dia{dialogue_id}_utt{utterance_id}.mp4")
        
        try: 
            cap = cv2.VideoCapture(video_file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []
            for i in range(0, total_frames, self.frames_to_sample):
                cap.set(1, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            embeddings = []
            for frame in frames:
                inputs = self.processor(frame, return_tensors="pt")
                embeddings.append(inputs['pixel_values'].squeeze())

            video_embedding = torch.mean(torch.stack(embeddings), dim=0)
        
            return {
                "pixel_values": video_embedding,
                "label": self.emotion_mapping[emotion]
            }
        except RuntimeError:
            print(f"File {video_file_path} is corrupted. Skipping...")
            # Recursively call the next item
            return self.__getitem__(idx + 1 if idx + 1 < len(self) else 0)


    
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

processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
train_dataset = MeldVideoDataset(csv_path="./MELD.Raw/train_sent_emo.csv", video_folder="./MELD.Raw/train_splits", processor=processor)
train_dataset = train_dataset
eval_dataset = MeldVideoDataset(csv_path="./MELD.Raw/test_sent_emo.csv", video_folder="./MELD.Raw/output_repeated_splits_test", processor=processor)
eval_dataset = eval_dataset
num_labels = len(train_dataset.emotion_mapping)

# Instantiate the VideoClassification model
# Use default config, or you can pass a custom config
# Initialize processor and model
config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224-pt22k', dropout=0.4, attention_dropout=0.4)
model = VideoClassification("microsoft/beit-base-patch16-224-pt22k", num_labels=num_labels, config=config)
model = model

training_args = TrainingArguments(
    output_dir="./results_meld_vid",
    evaluation_strategy="epoch",
    per_device_train_batch_size=64,  # Adjust batch size according to your GPU capacity
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # ... add other relevant arguments as needed
)

optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8) # lr=2e-5
num_training_steps = len(train_dataset) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=250,  # Can be changed based on preference
    num_training_steps=num_training_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    optimizers=(optimizer, lr_scheduler)
)

# 5. Start training
trainer.train()


# 6. Save your model
#torch.save(model.state_dict(), "./multimodal_mpc_call/wav2vec2_meld_pretrained/pytorch_model.bin")
torch.save(model.state_dict(), "./BEiT_meld_pretrained/pytorch_model.bin")
processor.save_pretrained("./BEiT_meld_pretrained")
