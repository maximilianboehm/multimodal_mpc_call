import os
import sys
import numpy as np
import pandas as pd
import traceback
from tqdm import tqdm
from dateutil import parser
from datetime import datetime, timedelta
from numbers_parser import Document

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

bank_target_dict = {
    'scripts_boc': "Canada.numbers",
    'scripts_boe': "England.numbers",
    'scripts_bonz': "New Zealand.numbers",
    'scripts_ecb': "Europe.numbers",
    'scripts_frb': "US.numbers",
    'scripts_bosa': "South Africa.numbers",
}

def exists(path):
    """
    Check if a file exists at the given path.

    Parameters:
    - path (str): The path to the file.

    Returns:
    - bool: True if a file exists at the specified path, False
        otherwise.
    """
    ans = os.path.isfile(path)
    return ans

def load_labels_df(path):
    """
    Load data from a spreadsheet file into a Pandas DataFrame.

    Parameters:
    - path (str): The path to the spreadsheet file.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the data from the 
        spreadsheet.
    """
    doc = Document(path)
    sheets = doc.sheets
    tables = sheets[0].tables
    rows = tables[0].rows(values_only=True)
    df = pd.DataFrame(rows).drop(columns=[0])
    df = df[1:]
    return df

def get_subset(df, target_var=1):
    """
    Extract a subset of a DataFrame containing date and target
    variable columns.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - target_var (int): The index of the target variable in the
        DataFrame.
        Default is 1.

    Returns:
    - df_sub (pandas.DataFrame): A subset of the input DataFrame
        containing date and target variable columns.
    """
    n = target_var
    target_col_idx = 2*n
    date_col_idx = target_col_idx - 1
    df_sub = df[[date_col_idx, target_col_idx]].dropna()
    df_sub = df_sub.rename(
        columns=df_sub.iloc[0]
    ).drop(df_sub.index[0]).reset_index(drop=True)
    return df_sub

def calculate_date_format(target_date, curr_date_parsed, ctry):
    """
    Calculate the date format based on the country code.

    Parameters:
    - target_date (datetime.datetime): The target date parsed as a
        datetime object.
    - curr_date_parsed (datetime.datetime): The current date parsed
        as a datetime object.
    - ctry (str): The country code or identifier.

    Returns:
    - str: The formatted target date.
    """

    if ctry in ["US.numbers", "Europe.numbers", "England.numbers"]:
        target_date = target_date.strftime("%-m/%-d/%Y")
        curr_date_parsed = curr_date_parsed.strftime("%-m/%-d/%Y")
    else:
        target_date = target_date.strftime("%Y-%m-%d")
        curr_date_parsed = curr_date_parsed.strftime("%Y-%m-%d")
    return target_date, curr_date_parsed


def get_movement_target(df_target, target_date, curr_date, parser):
    """
    Get the movement target for the given target and current dates.

    Parameters:
    - df_target (pandas.DataFrame): The DataFrame containing target
        variable data.
    - target_date (str): The formatted target date.
    - curr_date (str): The formatted current date.
    - parser (datetime.parser): The parser object.

    Returns:
    - tuple: A tuple containing the movement (1 for increase, 0 for
        decrease) and the parsed target date.
    """
    row_label_1 = df_target[df_target[df_target.columns[0]] == target_date]
    row_label_2 = df_target[df_target[df_target.columns[0]] == curr_date]

    if len(row_label_1) == 0 or len(row_label_2) == 0:
        return np.random.randint(0, 2), parser.parse(target_date)

    parsed_date = parser.parse(target_date)
    movement = int(row_label_1.values[0][1] > row_label_2.values[0][1])

    return movement, parsed_date


def calculate_volatility(df_target, target_date, volatility_window, parser):
    """
    Calculate the volatility for the given target date and window.

    Parameters:
    - df_target (pandas.DataFrame): The DataFrame containing target
        variable data.
    - target_date (str): The formatted target date.
    - volatility_window (int): The window size for calculating
        volatility.
    - parser (datetime.parser): The parser object.

    Returns:
    - tuple: A tuple containing the calculated volatility and the
        parsed target date.
    """
    row_label = df_target[df_target[df_target.columns[0]] == target_date]

    if len(row_label) == 0:
        return 0, parser.parse(target_date)

    target_index = (
        df_target[df_target[df_target.columns[0]] == target_date].index[0]
    )
    last_x_days_rows = (
        df_target.iloc[target_index:target_index + volatility_window]
    )

    last_x_days_rows = last_x_days_rows.copy()
    last_x_days_rows["Return"] = (
        last_x_days_rows[last_x_days_rows.columns[1]].pct_change()
    )
    last_x_days_rows = last_x_days_rows[1:]
    average_return = last_x_days_rows["Return"].mean()
    last_x_days_rows["Variance"] = (
        (last_x_days_rows["Return"] - average_return) ** 2
    )
    var_sum = last_x_days_rows["Variance"].sum()

    if var_sum / volatility_window == 0:
        volatility = 0
    else:
        volatility = np.log(np.sqrt(var_sum / volatility_window))

    return volatility, parser.parse(target_date)


def get_target(
    df_target,
    curr_date_parsed,
    offset,
    ctry,
    movement=False,
    volatility_window=1,
):
    """
    Get the target variable value and date for a given offset from the
    current date.

    Parameters:
    - df_target (pandas.DataFrame): The DataFrame containing target
        variable data.
    - curr_date_parsed (datetime.datetime): The current date parsed
        as a datetime object.
    - offset (int): The number of days offset from the current date.
    - ctry (str): The country code or identifier.
    - movement (bool): If True, calculate the movement between current
        and target dates. Default is False.
    - volatility_window (int): The window size for calculating
        volatility. Default is 1.

    Returns:
    - If movement is True:
        - movement (int): The movement between current and target dates 
            (1 for increase, 0 for decrease).
        - target_date (datetime.datetime): The target date.
    - If movement is False:
        - volatility (float): The calculated volatility.
        - target_date (datetime.datetime): The target date.
    """
    target_date = curr_date_parsed + timedelta(days=offset)
    target_date_formatted, curr_date_formatted = calculate_date_format(
        target_date, curr_date_parsed, ctry
    )

    if movement:
        return get_movement_target(
            df_target, target_date_formatted, curr_date_formatted, parser
        )
    else:
        return calculate_volatility(
            df_target, target_date_formatted, volatility_window, parser
        )
    
    
class MultimodalDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path="../", subclip_maxlen=-1):
        self.data_path = data_path
        self.sub_data = [
            'scripts_boc',
            'scripts_bonz',
            'scripts_ecb',
            'scripts_frb',
            'scripts_bosa']
        self.text = []
        self.audio = []
        self.video = []
        self.labels = []
        self.timestamps_dt = []
        self.timestamps = []
        self.subclip_mask = []
        self.subclip_maxlen = subclip_maxlen
    
    def load_data(
        self, data_path=None, offset=1, movement=False, volatility_window=1
    ):
    """
    Load data from CSV files and associated embeddings.

    Parameters:
    - data_path (str): The path to the directory containing data files.
        If None, it uses the default data_path.
    - offset (int): The number of days offset from the current date for
        calculating target variables. Default is 1.
    - movement (bool): If True, calculate the movement between current
        and target dates. Default is False.
    - volatility_window (int): The window size for calculating
        volatility. Default is 1.

    Returns:
    - None: The method modifies the internal state of the class
        instance.
    """
    errs = 0
    tot = 0
    
    if data_path is None:
        data_path = self.data_path
        
    for sub in tqdm(self.sub_data):
        self.load_sub_data(
            sub, data_path, offset, movement, volatility_window, errs, tot
        )
        
    print("SKIPPED: ", errs, " out of ", tot)
    print("DONE.")

    def load_sub_data(
        self, sub, data_path, offset, movement, volatility_window, errs, tot
    ):
        """
        Load data for a subcategory and process associated video data. 

        Parameters:
        - sub (str): The subcategory to load data for.
        - data_path (str): The path to the directory containing data
            files.
        - offset (int): The number of days offset from the current date
            for calculating target variables.
        - movement (bool): If True, calculate the movement between
            current and target dates.
        - volatility_window (int): The window size for calculating
            volatility.
        - errs (int): The current count of errors encountered.
        - tot (int): The total count of items processed.

        Returns:
        - None: The method modifies the internal state of the class
            instance.
        """
        print("Loading: ", sub)
        root_f = os.path.join(data_path, sub)
        csvfile = self.find_csv_file(root_f)
        df = pd.read_csv(
            os.path.join(root_folder, csvfile), header=None
        )
        root_folder = os.path.join(root_folder, "data")
        ctry = bank_target_dict[sub]
        df_target = load_labels_df(
            os.path.join(data_path, "price_data", ctry)
        )
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                tot += 1
                st = row.values[1]
                st = eval(st)
                bank = st['Bank']
                date = st['Date']
                video = st['Video_links']
                folder = date+'_'+bank+'_'+str(idx)
                folderpath = os.path.join(root_folder, folder)
                if not os.path.isdir(folderpath):
                    folder = date+'_'+bank
                    folderpath = os.path.join(root_folder, folder)
                parsed_date = parser.parse(date)
                labels = self.get_labels(
                    df_target,
                    parsed_date,
                    offset,
                    ctry,
                    movement,
                    volatility_window
                )
                self.process_video_data(folderpath, labels, folder, errs)
            except:
                errs += 1
        print("ERRORS CURRENT: ", errs)

    def find_csv_file(self, root_f):
        """
        Find the CSV file in the specified directory.

        Parameters:
        - root_f (str): The path to the directory where the CSV file is
            located.

        Returns:
        - str: The path of the CSV file found in the directory.
        """
        p = [each for each in os.listdir(root_f) if each.endswith('.csv')][0]
        return p

    def get_labels(
        self,
        df_target,
        parsed_date,
        offset,
        ctry,
        movement,
        volatility_window
    ):
        """
        Get labels for the dataset based on target variables.

        Parameters:
        - df_target (pandas.DataFrame): The DataFrame containing target
            variable data.
        - parsed_date (datetime.datetime): The current date parsed as a
            datetime object.
        - offset (int): The number of days offset from the current
            date.
        - ctry (str): The country code or identifier.
        - movement (bool): If True, calculate the movement between
            current and target dates. Default is False.
        - volatility_window (int): The window size for calculating
            volatility. Default is 1.

        Returns:
        - list: A list of labels corresponding to each subset of the
            target data.
        """
        labels = []
        for i in range(6):
            df_subset = get_subset(df_target, i+1)
            label, timestamp = get_target(
                df_subset,
                parsed_date,
                offset,
                ctry,
                movement,
                volatility_window
            )
            labels.append(label)
        return labels

    def process_video_data(self, folderpath, labels, folder, errs):
        """
        Process video data and append it to the dataset.

        Parameters:
        - folderpath (str): The path to the folder containing the video
            data.
        - labels (list): List of labels.
        - folder (str): The folder name.
        - errs (int): The current count of errors.

        Returns:
        - None: The method modifies the internal state of the class
            instance.
        """
        videofile = os.path.join(folderpath, "video_fragments.npz")
        transcriptfile = os.path.join(folderpath, "fin_bert_embeddings.npz")
        audiofile = os.path.join(
            folderpath, "wav2_vec2_finetuned_embeddings.npz"
        )
        if exists(videofile) and exists(transcriptfile) and exists(audiofile):
            video_embs = self.load_video_embeddings(videofile)
            bert_embs = np.load(transcriptfile, allow_pickle=True)['arr_0']
            wav2vec2_embs = np.load(audiofile, allow_pickle=True)['arr_0']
            self.append_data(video_embs, bert_embs, wav2vec2_embs, labels)
        else:
            self.log_error(folderpath, errs)

    def load_video_embeddings(self, videofile):
        """
        Load video embeddings from a file and preprocess them.

        Parameters:
        - videofile (str): The path to the file containing video
            embeddings.

        Returns:
        - list: A list of preprocessed video embeddings as torch
            tensors.
        """
        video_embs = np.load(videofile, allow_pickle=True)['arr_0']
        if self.subclip_maxlen == -1:
            video_embs = [
                np.mean(x, axis=0, keepdims=True) for x in video_embs
            ]
            self.subclip_maxlen = 1
        s_mask = np.zeros((len(video_embs), self.subclip_maxlen))
        for idx,vid in enumerate(video_embs):
            s_mask[idx,:len(vid)] = 1
        video_embs = [
            nn.ZeroPad2d((0, 0, 0, self.subclip_maxlen - len(x)))(
                torch.tensor(x, dtype=torch.float))
            for x in video_embs
        ]
        return video_embs

    def append_data(self, video_embs, bert_embs, wav2vec2_embs, labels):
        """
        Append data to the corresponding lists.

        Parameters:
        - video_embs (list): List of video embeddings.
        - bert_embs (list): List of BERT embeddings.
        - wav2vec2_embs (list): List of Wav2Vec2 embeddings.
        - labels (list): List of labels.

        Returns:
        - None: The method modifies the internal state of the class
            instance.
        """
        self.video.append(torch.stack(video_embs))
        self.subclip_mask.append(s_mask)
        self.text.append(bert_embs)
        self.audio.append(wav2vec2_embs)
        self.labels.append(np.array(labels))
        self.timestamps_dt.append((timestamp, videofile))
        self.timestamps.append(datetime.timestamp(timestamp))

    def log_error(self, folderpath, errs):
         """
        Log an error to a text file.

        Parameters:
        - folderpath (str): The path of the folder where the error
            occurred.
        - errs (int): The current count of errors.

        Returns:
        - int: The updated count of errors.
        """
        with open('errors_dataloader.txt', 'a') as f:
            print("Error 1")
            f.write(str(folderpath)+"\n")
        errs += 1    
        
        
    def make_splits(self, ratios = [0.7, 0.1, 0.2]):
        """
        Make train, validation, and test splits based on timestamps.
    
        Parameters:
        - ratios (list): A list containing the ratios for train,
                         validation, and test sets respectively.
                         Default is [0.7, 0.1, 0.2].
    
        Returns:
        - train_idx (numpy.ndarray): An array of indices for the
            train set.
        - val_idx (numpy.ndarray): An array of indices for the
            validation set.
        - test_idx (numpy.ndarray): An array of indices for the
            test set.
        """
        indices = np.argsort(np.array(self.timestamps))
        n = len(indices)
        edges = [0, int(ratios[0]*n), int((ratios[0]+ratios[1])*n),n]
        train_idx = indices[edges[0]:edges[1]]
        val_idx = indices[edges[1]:edges[2]]
        test_idx = indices[edges[2]:edges[3]]
        return train_idx, val_idx, test_idx
    
    def __len__(self):
        """
        Get the length of the dataset.
    
        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.text)
    
    def __getitem__(self, idx):
        """
        Retrieve data at the specified index.
    
        Parameters:
        - idx (int): The index of the data to retrieve.
    
        Returns:
        - label: The label associated with the data.
        - video: The video data.
        - audio: The audio data as a PyTorch tensor with dtype
            torch.float.
        - text: The text data as a PyTorch tensor with dtype
            torch.float.
        - subclip_mask: The subclip mask as a PyTorch tensor with 
            dtype torch.bool.
        - timestamp: The timestamp associated with the data.
        """
        label = self.labels[idx]
        video = self.video[idx]
        audio = torch.tensor(self.audio[idx], dtype=torch.float)
        text = torch.tensor(self.text[idx], dtype=torch.float)
        subclip_mask = torch.tensor(
            self.subclip_mask[idx], dtype=torch.bool)
        timestamp = self.timestamps_dt[idx]
        
        return label, video, audio, text, subclip_mask, timestamp
        
