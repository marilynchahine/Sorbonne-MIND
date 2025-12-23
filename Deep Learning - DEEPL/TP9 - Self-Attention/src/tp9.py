
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import time
import logging
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

GLOVE_PATH = Path("data/glove")
DATASET_PATH = Path("data/aclImdb")
IMDB_CLASSES  = ['neg','pos']


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")
    glove_fn = open(GLOVE_PATH / ("glove.6B.%dd.txt" % embedding_size))
    words, embeddings = [], []
    for line in glove_fn:
        values = line.split()
        words.append(values[0])
        embeddings.append([float(x) for x in values[1:]])

    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")


    return word2id, embeddings, FolderText(IMDB_CLASSES, DATASET_PATH /"train", tokenizer, load=False), FolderText(IMDB_CLASSES, DATASET_PATH / "test", tokenizer, load=False)




MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

#  TODO: 
    ##  TODO: 


if __name__ == "__main__":
    main()

# [[/STUDENT]]
