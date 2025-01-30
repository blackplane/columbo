# https://medium.com/@drjulija/llm-guardrails-how-i-built-a-toxic-content-classifier-4d9ecb9636ba

from pathlib import Path
from typing import Callable
import wandb
import pandas as pd
from matplotlib import pyplot as plt
from mpmath.identification import transforms
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from functools import partial


def build_embedding(model_path: str = 'Alibaba-NLP/gte-base-en-v1.5', device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return partial(tokenizer, max_length=8192, padding=True, truncation=True, return_tensors='pt'), \
            AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)


def embed(text: str, tokenizer: Callable, embedder: Callable, device: str = "cpu", normalize=True):
    batch_dict = tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = embedder(**batch_dict)
    del batch_dict
    embedding = outputs.last_hidden_state[:, 0]
    if normalize:
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().detach().numpy()


def get_datasets(device: str = "cpu", tokenizer: Callable = None, embedder: Callable = None, path: Path = Path.home() / "Data" / "Wikipedia-Toxic-Comments"):
    train_ds = WikipediaToxicCommentsDataset(path / "balanced_train.csv", tokenizer=tokenizer, embedder=embedder, device=device)
    val_ds = WikipediaToxicCommentsDataset(path / "validation.csv", tokenizer=tokenizer, embedder=embedder, device=device)
    test_ds = WikipediaToxicCommentsDataset(path / "test.csv", tokenizer=tokenizer, embedder=embedder, device=device)
    return train_ds, val_ds, test_ds


class  WikipediaToxicCommentsDataset(Dataset):
    """A Dataset class for the Wikipedia Toxic Comments Dataset"""
    def __init__(self, dataset_file: Path = None, tokenizer: Callable = None, embedder: Callable = None, device: str = "cpu"):
        if not (dataset_file.exists() and dataset_file.is_file()):
            dataset_file = Path.home() / "Data" / "Wikipedia-Toxic-Comments" / "balanced_train.csv"
        self._dataset = pd.read_csv(dataset_file)
        if not tokenizer and not embedder:
            tokenizer, embedder = build_embedding(device=device)
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.device = device

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        _, x, label = self._dataset.iloc[idx]
        if self.tokenizer:
            x = self.tokenizer(x).to(self.device)
        if self.embedder:
            x = self.embedder(**x)
            x = x.last_hidden_state[:, 0]
        label = torch.tensor(label).to(self.device)
        return {
            "embedding": x, "label": label
        }

