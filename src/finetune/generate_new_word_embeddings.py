import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

from finbert_embedding.embedding import FinbertEmbedding

# Read the paths to valid embedding folders from a CSV file
valid_paths = pd.read_csv("valid_embedding_folders.csv")

for path in tqdm(valid_paths):
    # Open the syncmap JSON file to retrieve synchronization information
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    
    embeddings_list = []
    
    # Loop through each fragment in the syncmap
    for fragment in syncmap['fragments']:
        # Extract text from the fragment
        text = fragment['lines'][0]
        # Initialize Finbert embedding model
        finbert = FinbertEmbedding()
        # Calculate sentence embedding using Finbert
        sentence_embedding = finbert.sentence_vector(text)
        embeddings_list.append(sentence_embedding)
    
    # Stack the embeddings into an array
    embeddings_array = np.vstack(embeddings_list)
    
    # Save the embeddings array to an NPZ file
    np.savez(os.path.join(path, 'fin_bert_embeddings.npz'), arr_0=embeddings_array)

        