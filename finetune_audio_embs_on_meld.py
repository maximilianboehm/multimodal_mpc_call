import os
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchaudio
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config

from sklearn.metrics import f1_score

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, random_split
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

class MeldAudioDataset(Dataset):
    def __init__(self, csv_path, audio_folder, processor, sample_rate=16000):
        """
        Args:
        - csv_path: Path to the .csv file.
        - audio_folder: Path to "train_splits" folder containing audio clips.
        - tokenizer: Initialized Wav2Vec2Tokenizer.
        - sample_rate: Desired sample rate for the audio files.
        """
        
        # Read the csv
        self.dataframe = pd.read_csv(csv_path)
        
        # Store other information
        self.audio_folder = audio_folder
        self.processor = processor
        self.sample_rate = sample_rate
        
        # Create a mapping for emotions to integers (labels)
        self.emotion_mapping = {emotion: i for i, emotion in enumerate(self.dataframe["Emotion"].unique())}
        
    def __len__(self):
        """Return dataset size"""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """Load audio, tokenize, and return with label"""
        
        # Extract relevant info from the dataframe
        dialogue_id = self.dataframe.iloc[idx]["Dialogue_ID"]
        utterance_id = self.dataframe.iloc[idx]["Utterance_ID"]
        emotion = self.dataframe.iloc[idx]["Emotion"]
        
        # Construct file path using dialogue_id and utterance_id
        audio_file_path = os.path.join(self.audio_folder, f"dia{dialogue_id}_utt{utterance_id}.mp4")
        
        try:
            # Load the audio. Note: torchaudio's mp4 support might need ffmpeg.
            waveform, sample_rate = torchaudio.load(audio_file_path)

            # If the sample rate is not the desired rate, resample it.
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.mean(dim=0)
            # Tokenize the waveform for wav2vec2
            #tokens = self.processor(waveform.squeeze().numpy(), return_tensors="pt", padding="longest", sampling_rate=self.sample_rate).input_values
            tokens = self.processor(
                waveform.squeeze().numpy(), 
                return_tensors="pt", 
                padding="max_length", 
                max_length=30000, 
                truncation=True, 
                sampling_rate=self.sample_rate
            ).input_values

            tokens = tokens.squeeze(0)

            # Return the tokenized audio data and the emotion label as an integer
            return {
                "input_values": tokens,  # Batch size is 1 here due to no batching in Dataset
                "label": self.emotion_mapping[emotion]
            }
    
        except RuntimeError:
            print(f"File {audio_file_path} is corrupted. Skipping...")
            # Recursively call the next item
            return self.__getitem__(idx + 1 if idx + 1 < len(self) else 0)
    
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
        # Here, mean pooling
        
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

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

train_dataset = MeldAudioDataset(csv_path="./MELD.Raw/train_sent_emo.csv", audio_folder="./MELD.Raw/train_splits", processor=processor)
eval_dataset = MeldAudioDataset(csv_path="./MELD.Raw/test_sent_emo.csv", audio_folder="./MELD.Raw/output_repeated_splits_test", processor=processor)

num_labels = len(train_dataset.emotion_mapping)
config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base-960h', dropout=0.5, attention_dropout=0.5) 
model = Wav2Vec2ForClassification("facebook/wav2vec2-base-960h", num_labels, config=config)


# 4. Training setup
training_args = TrainingArguments(
    output_dir="./results_meld",
    evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=512,  # Adjust batch size according to your GPU capacity
    per_device_eval_batch_size=512,
    num_train_epochs=5,
    weight_decay=0.01, # 0.01
    logging_dir='./logs',
    logging_steps=2,
    #warmup_steps=1000,
    # lr_scheduler_type="constant",
)

optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8) # lr=2e-5
num_training_steps = len(train_dataset) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100,  # Can be changed based on preference
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


# 6. Save model
torch.save(model.state_dict(), "./wav2vec2_meld_pretrained/pytorch_model.bin")
processor.save_pretrained("./wav2vec2_meld_pretrained")
