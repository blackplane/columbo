# https://medium.com/@drjulija/llm-guardrails-how-i-built-a-toxic-content-classifier-4d9ecb9636ba

from pathlib import Path
from typing import Callable
import toml
import numpy as np
import wandb
import pandas as pd
from matplotlib import pyplot as plt
from mpmath.identification import transforms
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from functools import partial


def build_embedding(model_path: str = 'Alibaba-NLP/gte-base-en-v1.5', device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return partial(tokenizer, max_length=8192, padding=True, truncation=True, return_tensors='pt'), \
            AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

# https://discuss.huggingface.co/t/bert-embedding-on-gpu/93448

def embed(text: str, tokenizer: Callable, embedder: Callable, device: str = "cpu", normalize=True):
    batch_dict = tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = embedder(**batch_dict)
    del batch_dict
    embedding = outputs.last_hidden_state[:, 0]
    if normalize:
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().detach().numpy()


def get_datasets_with_embedding(device: str = "cpu", path: Path = Path("..") / "Data" / "Wikipedia-Toxic-Comments"):
    paths = ["balanced_train_with_clean_embeddings.parquet.gzip", "validation_with_clean_embeddings.parquet.gzip", "test_with_clean_embeddings.parquet.gzip"]
    paths = [path / p for p in paths]
    datasets = [WikipediaToxicCommentsWithEmbeddingsDataset(path, device) for path in tqdm(paths)]
    train_ds, val_ds, test_ds = datasets
    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }


class ToxicityClassifierV1(nn.Module):
    ARCH = "MLP"
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super(ToxicityClassifierV1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class ToxicityClassifierV2(nn.Module):
    ARCH = "CNN"
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super(ToxicityClassifierV2, self).__init__()
        pool_kernel_size = pool_stride = 3
        pool_output_size = (input_size - pool_kernel_size) // pool_stride + 1
        self.model = nn.Sequential(
            # nn.Conv1d(input_size, hidden_size, 3, padding=1),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.MaxPool1d(kernel_size=3),
            nn.Linear(pool_output_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class ToxicityClassifierV3(nn.Module):
    ARCH = "MLP"
    """Just a linear layer, equuals logistic regression."""
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super(ToxicityClassifierV3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class  WikipediaToxicCommentsWithEmbeddingsDataset(Dataset):
    """A Dataset class for the Wikipedia Toxic Comments Dataset with embeddings."""
    def __init__(self, dataset_file, device: str = "cpu"):
        if isinstance(dataset_file, str):
            dataset_file = Path(dataset_file)
        self._dataset = pd.read_parquet(dataset_file)
        embeddings = [np.array(list(np.float32(y) for y in x[2:-2].split())) for x in self._dataset["embeddings"]]
        self._dataset["embeddings"] = embeddings
        self.device = device

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        row_id, comment_text, label, embedding = self._dataset.iloc[idx]
        embedding = torch.tensor(embedding).to(self.device)
        label = torch.tensor(label).to(self.device)
        return embedding.unsqueeze(dim=1), label
