import os
import pandas as pd
import cv2
import torch
from transformers import (
    BeitImageProcessor, BeitForImageClassification, BeitConfig, BeitModel
)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, random_split
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# Use cuda if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class MeldVideoDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_folder,
        processor,
        frames_to_sample=10
    ):
        self.dataframe = pd.read_csv(csv_path)
        self.video_folder = video_folder
        self.processor = processor
        self.frames_to_sample = frames_to_sample
        self.emotion_mapping = {
            emotion: i for i, emotion in enumerate(
                self.dataframe["Emotion"].unique()
            )
        }
        
    def __len__(self):
        """
        Get the length of the dataset.
    
        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.dataframe)
    
    def extract_info(self, idx):
    """
    Extract relevant information from the dataset at the specified
    index.

    Parameters:
    - idx (int): The index of the sample to retrieve.

    Returns:
    - tuple: A tuple containing dialogue_id, utterance_id, and emotion.
    """
    dialogue_id = self.dataframe.iloc[idx]["Dialogue_ID"]
    utterance_id = self.dataframe.iloc[idx]["Utterance_ID"]
    emotion = self.dataframe.iloc[idx]["Emotion"]
    return dialogue_id, utterance_id, emotion


def construct_video_file_path(self, dialogue_id, utterance_id):
    """
    Construct the file path for the video based on dialogue_id and
    utterance_id.

    Parameters:
    - dialogue_id (str): The dialogue ID.
    - utterance_id (str): The utterance ID.

    Returns:
    - str: The constructed video file path.
    """
    return os.path.join(
        self.video_folder, f"dia{dialogue_id}_utt{utterance_id}.mp4"
    )


def load_video_embeddings(self, video_file_path):
    """
    Load and process video embeddings from the specified video file
    path.

    Parameters:
    - video_file_path (str): The file path of the video.

    Returns:
    - torch.Tensor: The video embeddings represented as a PyTorch
        tensor.
    """
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
    return video_embedding

def handle_corrupted_file(self, idx):
    """
    Handles a corrupted audio file by skipping it and moving to the
    next item.

    Parameters:
    - idx (int): The index of the corrupted audio file.

    Returns:
    - dict: The next item in the dataset.
    """
    print(f"File {audio_file_path} is corrupted. Skipping...")
    return self.__getitem__(idx + 1 if idx + 1 < len(self) else 0)

def __getitem__(self, idx):
    """
    Retrieve a sample from the dataset at the specified index.

    Parameters:
    - idx (int): The index of the sample to retrieve.

    Returns:
    - dict: A dictionary containing the following keys:
        - "pixel_values": The video embedding represented as a
            PyTorch tensor.
        - "label": The index of the emotion label mapped from the
            dataset.
    """
    dialogue_id, utterance_id, emotion = self.extract_info(idx)
    video_file_path = self.construct_video_file_path(
        dialogue_id, utterance_id
    )
    
    try: 
        video_embedding = self.load_video_embeddings(video_file_path)
        return {
            "pixel_values": video_embedding,
            "label": self.emotion_mapping[emotion]
        }
    except RuntimeError:
        return self.handle_corrupted_file(idx)

    
class VideoClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, config):
        super(VideoClassification, self).__init__()
        self.beit = BeitModel.from_pretrained(
            pretrained_model_name,
            config=config
        )
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.beit.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        # Add classification layer
        self.classifier = nn.Linear(
            self.beit.config.hidden_size,
            num_labels
        )
        
    def forward(self, pixel_values, labels=None):
        """
        Forward pass through the model.

        Parameters:
        - pixel_values (torch.Tensor): The input pixel values of
            images.
        - labels (torch.Tensor or None): The target labels for
            classification. Default is None.
    
        Returns:
        - If labels are provided:
            - loss (torch.Tensor): The computed loss using
                cross-entropy.
            - logits (torch.Tensor): The output logits from the
                classifier.
        - If labels are not provided:
            - logits (torch.Tensor): The output logits from the
                classifier.
        """
        # Extract features from images
        outputs = self.beit(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        
         # Compute attention weights
        attention_weights = self.attention(features)
        
        # Apply attention weights to features
        attended_features = torch.sum(
            attention_weights * features,
            dim=1
        )

        #pooled = features.mean(dim=1)

        # Classification layer
        logits = self.classifier(attended_features) 
        
        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return (loss, logits) if loss is not None else logits

processor = BeitImageProcessor.from_pretrained(
    "microsoft/beit-base-patch16-224-pt22k"
)
train_dataset = MeldVideoDataset(
    csv_path="./MELD.Raw/train_sent_emo.csv",
    video_folder="./MELD.Raw/train_splits",
    processor=processor)
train_dataset = train_dataset
eval_dataset = MeldVideoDataset(
    csv_path="./MELD.Raw/test_sent_emo.csv",
    video_folder="./MELD.Raw/output_repeated_splits_test",
    processor=processor)
eval_dataset = eval_dataset
num_labels = len(train_dataset.emotion_mapping)

# Instantiate the VideoClassification model
# Use default config, or pass a custom config
# Initialize processor and model
config = BeitConfig.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k',
    dropout=0.4,
    attention_dropout=0.4
)
model = VideoClassification(
    "microsoft/beit-base-patch16-224-pt22k",
    num_labels=num_labels,
    config=config
)
model = model

# Training setup
training_args = TrainingArguments(
    output_dir="./results_meld_vid",
    evaluation_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Init optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8)
num_training_steps = (
    len(train_dataset) * training_args.num_train_epochs
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=250, 
    num_training_steps=num_training_steps
)

# Init Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    optimizers=(optimizer, lr_scheduler)
)

# Start training
trainer.train()


# Save model
torch.save(model.state_dict(), "./BEiT_meld_pretrained/pytorch_model.bin")
processor.save_pretrained("./BEiT_meld_pretrained")
