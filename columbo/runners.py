from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from columbo.toxicity import embed, build_embedding, ToxicityClassifierV1, get_datasets_with_embedding
import logging
import friendlywords as fw
from codetiming import Timer
import wandb


logger = logging.getLogger(__name__)


def get_device(device):
    if device is not None:
        pass
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device={device}")
    return device


def run_training(epochs:int=10, device=None):
    device = get_device(device)

    logger.info("Loading datasets ...")
    datasets = get_datasets_with_embedding(device, Path() / "Data" / "Wikipedia-Toxic-Comments")
    dataloaders = {
        k: DataLoader(
            ds,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        for k, ds in datasets.items()
    }
    batch_size, input_dim = next(iter(dataloaders["train"]))["embedding"].shape
    model = ToxicityClassifierV1(input_size=input_dim, hidden_size=256, output_size=2).to(device)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    epoch_timer = Timer(name="epoch_timer", logger=None)
    batch_timer = Timer(name="batch_timer", logger=None)

    wandb.init(
        # set the wandb project where this run will be logged
        project="toxicity-classifier",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MLP",
            "dataset": dataloaders["train"].dataset.__class__,
            "epochs": epochs,
        }
    )

    for epoch in range(epochs):
        # Training phase
        epoch_timer.start()
        val_loss = -1.
        model.train()
        logger.info(f"---- Starting epoch {epoch} [last={epoch_timer.last:.2f} s | val_loss={val_loss:.4f}] ----")
        for idx, sample in enumerate(dataloaders["train"]):
            batch_timer.start()
            X, y = sample["embedding"], sample["label"]
            optimizer.zero_grad()
            y_pred = model(X)
            y_oh = torch.nn.functional.one_hot(y, num_classes=2).float()
            loss = loss_fn(y_pred.squeeze(), y_oh)
            loss.backward()
            optimizer.step()
            loss_np = loss.cpu().detach().numpy()
            losses.append(loss_np)
            wandb.log({"loss": loss_np, "loss_ma12": np.mean(losses[-12:]), "loss_ma32": np.mean(losses[-32:])})
            batch_timer.stop()
            if idx % 50 == 0:
                logger.info(f"{idx:5} Loss = {loss_np}")
        epoch_timer.stop()
        # Validation phase
        model.eval()
        with torch.no_grad():
            for sample in dataloaders["val"]:
                X, y = sample["embedding"], sample["label"]
                y_pred = model(X)
                y_oh = torch.nn.functional.one_hot(y, num_classes=2).float()
                loss = loss_fn(y_pred.squeeze(), y_oh)
                val_loss +=  loss.cpu().detach().numpy()

        val_loss /= len(dataloaders["val"])
        wandb.log({"val_loss_avg": val_loss})

    wandb.finish()

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        },
        f"toxicity_classifier_{fw.generate('po', separator='-')}.pth"
    )
    logger.info("DONE.")





def run_embeddings(src_csv, dest_csv, device=None):
    device = get_device(device)
    logger.info(f"Using device={device}")

    tokenizer, embedder = build_embedding(device=device)
    # dataset_dir = Path.cwd() / "Data" / "Wikipedia-Toxic-Comments"
    df = pd.read_csv(src_csv)

    with logging_redirect_tqdm():
        embeddings = [embed(text, tokenizer, embedder, device) for text in tqdm(df["comment_text"].values)]

    logger.info(f"Storing embeddings in file {dest_csv}")
    df["embeddings"] = embeddings
    df.to_parquet('dest_csv', compression='gzip')
