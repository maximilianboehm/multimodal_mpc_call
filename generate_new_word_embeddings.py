import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

from finbert_embedding.embedding import FinbertEmbedding

valid_paths = pd.read_csv("valid_embedding_folders.csv")
for path in tqdm(valid_paths):
    with open(os.path.join(path, "syncmap.json")) as f:
        syncmap = json.load(f)
    embeddings_list = []
    for fragment in syncmap['fragments']:
        #print(type(fragment['lines'][0]))
        text = fragment['lines'][0]
        finbert = FinbertEmbedding()
        sentence_embedding = finbert.sentence_vector(text)
        embeddings_list.append(sentence_embedding)
    embeddings_array = np.vstack(embeddings_list)
    np.savez(os.path.join(path, 'fin_bert_embeddings.npz'), arr_0=embeddings_array)
        