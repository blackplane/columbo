from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from columbo.toxicity import embed, build_embedding
import logging


logger = logging.getLogger(__name__)


def run_embeddings(src_csv, dest_csv, device=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device={device}")

    tokenizer, embedder = build_embedding(device=device)
    # dataset_dir = Path.cwd() / "Data" / "Wikipedia-Toxic-Comments"
    df = pd.read_csv(src_csv)

    with logging_redirect_tqdm():
        embeddings = [embed(text, tokenizer, embedder, device) for text in tqdm(df["comment_text"].values)]

    logger.info(f"Storing embeddings in file {dest_csv}")
    df["embeddings"] = embeddings
    df.to_parquet('dest_csv', compression='gzip')
