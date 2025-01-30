from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from columbo.toxicity import embed, build_embedding
import logging


logger = logging.getLogger(__name__)


def run_embeddings(src_csv="balanced_train.csv", dest_csv="balanced_train_with_embeddings.csv", device=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device("gpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    tokenizer, embedder = build_embedding(device=device)
    dataset_dir = Path.cwd() / "Data" / "Wikipedia-Toxic-Comments"
    df = pd.read_csv(dataset_dir / src_csv)

    with logging_redirect_tqdm():
        embeddings = [embed(text, tokenizer, embedder) for text in tqdm(df["comment_text"].values)]
    embeddings = [e.cpu().detach().numpy() for e in embeddings]
    df["embeddings"] = embeddings

    df.to_csv(dataset_dir / dest_csv)
