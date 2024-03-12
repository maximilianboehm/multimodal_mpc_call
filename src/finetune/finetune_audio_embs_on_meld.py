import os
import pandas as pd
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
        - audio_folder: Path to "train_splits" folder containing audio
            clips.
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
        self.emotion_mapping = {
            emotion: i for i, emotion in enumerate(
                self.dataframe["Emotion"].unique())
            }
        
    def __len__(self):
        """
        Get the length of the DataFrame stored within the object.

        Returns:
        - int: The number of rows in the DataFrame.
        """
        return len(self.dataframe)
    
    def extract_info(self, idx):
        """
        Extracts relevant information from the dataframe at the
        specified index.

        Parameters:
        - idx (int): The index of the row in the dataframe.

        Returns:
        - tuple: A tuple containing the extracted information:
            - dialogue_id (str): The ID of the dialogue.
            - utterance_id (str): The ID of the utterance.
            - emotion (str): The emotion associated with the dialogue
                utterance.
        """
        dialogue_id = self.dataframe.iloc[idx]["Dialogue_ID"]
        utterance_id = self.dataframe.iloc[idx]["Utterance_ID"]
        emotion = self.dataframe.iloc[idx]["Emotion"]
        return dialogue_id, utterance_id, emotion

    def construct_file_path(self, dialogue_id, utterance_id):
        """
        Constructs the file path for the audio file based on the
        dialogue ID and utterance ID.

        Parameters:
        - dialogue_id (str): The ID of the dialogue.
        - utterance_id (str): The ID of the utterance.

        Returns:
        - str: The file path for the audio file.
        """
        return os.path.join(
            self.audio_folder, f"dia{dialogue_id}_utt{utterance_id}.mp4"
        )

    def load_audio(self, audio_file_path):
        """
        Loads the audio waveform from the specified file path.

        Parameters:
        - audio_file_path (str): The file path of the audio file.

        Returns:
        - torch.Tensor: The audio waveform.
        """
        waveform, sample_rate = torchaudio.load(audio_file_path)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        return waveform

    def tokenize_audio(self, waveform):
        """
        Tokenizes the audio waveform for Wav2Vec2 model.

        Parameters:
        - waveform (torch.Tensor): The audio waveform to tokenize.

        Returns:
        - torch.Tensor: The tokenized audio data.
        """
        waveform = waveform.mean(dim=0)
        tokens = self.processor(
            waveform.squeeze().numpy(),
            return_tensors="pt",
            padding="max_length",
            max_length=30000,
            truncation=True,
            sampling_rate=self.sample_rate,
        ).input_values
        tokens = tokens.squeeze(0)
        return tokens

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
        Retrieves an item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the tokenized audio data as
            "input_values" and the emotion label as "label".
        """
        dialogue_id, utterance_id, emotion = self.extract_info(idx)
        audio_file_path = self.construct_file_path(dialogue_id, utterance_id)
        try:
            waveform = self.load_audio(audio_file_path)
            tokens = self.tokenize_audio(waveform)
            return {
                "input_values": tokens,
                "label": self.emotion_mapping[emotion]
            }
        except RuntimeError:
            return self.handle_corrupted_file(idx)

    
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
        Forward pass of the model.

        Parameters:
        - input_values (torch.Tensor): The tokenized input audio data.
        - labels (torch.Tensor or None): The ground truth emotion labels.
        
        Returns:
        - torch.Tensor or tuple: If labels are provided, returns a tuple containing
        the loss and the logits. Otherwise, returns just the logits.
        """
        # Extract features from audio
        features = self.wav2vec2(input_values).last_hidden_state
        
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

train_dataset = MeldAudioDataset(
    csv_path="./MELD.Raw/train_sent_emo.csv",
    audio_folder="./MELD.Raw/train_splits",
    processor=processor
)
eval_dataset = MeldAudioDataset(
    csv_path="./MELD.Raw/test_sent_emo.csv",
    audio_folder="./MELD.Raw/output_repeated_splits_test",
    processor=processor
)

num_labels = len(train_dataset.emotion_mapping)
config = Wav2Vec2Config.from_pretrained(
    'facebook/wav2vec2-base-960h',
    dropout=0.5,
    attention_dropout=0.5
) 
model = Wav2Vec2ForClassification(
    "facebook/wav2vec2-base-960h",
    num_labels,
    config=config
)


# Training setup
training_args = TrainingArguments(
    output_dir="./results_meld",
    evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=2,
    #warmup_steps=1000,
    # lr_scheduler_type="constant",
)

# Init optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
num_training_steps = len(train_dataset) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100,  # Can be changed based on preference
    num_training_steps=num_training_steps
)

# Init trainer
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
torch.save(model.state_dict(), "./wav2vec2_meld_pretrained/pytorch_model.bin")
processor.save_pretrained("./wav2vec2_meld_pretrained")
