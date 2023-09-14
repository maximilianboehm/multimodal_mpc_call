import os
import pandas as pd
import torchaudio
from transformers import Wav2Vec2Tokenizer
from torch.utils.data import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader, random_split

class MeldAudioDataset(Dataset):
    def __init__(self, csv_path, audio_folder, tokenizer: Wav2Vec2Tokenizer, sample_rate=16000):
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
        self.tokenizer = tokenizer
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
        
        # Load the audio. Note: torchaudio's mp4 support might need ffmpeg.
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # If the sample rate is not the desired rate, resample it.
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Tokenize the waveform for wav2vec2
        tokens = self.tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")
        
        # Return the tokenized audio data and the emotion label as an integer
        return {
            "input_values": tokens["input_values"][0],  # Batch size is 1 here due to no batching in Dataset
            "attention_mask": tokens["attention_mask"][0],  # Similarly, take the 0-th index
            "label": self.emotion_mapping[emotion]
        }


# 1. Initialize Tokenizer and Model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.lm_head = torch.nn.Linear(model.config.hidden_size, len(MeldAudioDataset.emotion_mapping))  # Adjust the number of output labels

# 2. Create dataset instance
train_dataset = MeldAudioDataset(csv_path="path_to_your_csv.csv", audio_folder="path_to_train_splits", tokenizer=tokenizer)
eval_dataset = MeldAudioDataset(csv_path="path_to_your_csv.csv", audio_folder="path_to_train_splits", tokenizer=tokenizer)

# 4. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Adjust batch size according to your GPU capacity
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # ... add other relevant arguments as needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer  # Added tokenizer which is necessary for Trainer to process batches
)

# 5. Start training
trainer.train()

# 6. Save your model
model.save_pretrained("path_to_save_directory")
tokenizer.save_pretrained("path_to_save_directory")
