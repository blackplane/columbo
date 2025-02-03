import string
from pathlib import Path
from random import choices
from typing import Optional, Union, Sequence, Tuple, Mapping, Callable
from rich.logging import RichHandler
import torch
import torch.nn as nn
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from columbo.toxicity import embed, build_embedding, ToxicityClassifierV1, get_datasets_with_embedding, \
    ToxicityClassifierV2, ToxicityClassifierV3, ToxicityClassifierV4
import logging
import friendlywords as fw
from codetiming import Timer
import wandb
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import ConfusionMatrix, Accuracy, Recall, Precision, MetricsLambda
from columbo.utils import preprocess


logger = logging.getLogger(__name__)

CLASSIFIER_MAP = {
    "v1": ToxicityClassifierV1,
    "v2": ToxicityClassifierV2,
    "v3": ToxicityClassifierV3,
    "v4": ToxicityClassifierV4,
}

def get_classifier(classifier: Union[str, Callable]=None):
    if not classifier:
        classifier = ToxicityClassifierV3
    elif isinstance(classifier, str):
        assert classifier in CLASSIFIER_MAP, f"Classifier {classifier} not part of CLASSIFIER_MAP={CLASSIFIER_MAP}."
        classifier = CLASSIFIER_MAP[classifier]
    return classifier


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


def run_training(epochs:int=10, classifier: Union[str, Callable]=None, device=None, run_id=None):
    device = get_device(device)
    classifier = get_classifier(classifier)

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
    a = next(iter(dataloaders["val"]))[0]
    batch_size, _, input_dim = next(iter(dataloaders["train"]))[0].shape
    model = classifier(input_size=input_dim, hidden_size=256, output_size=2).to(device)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    epoch_timer = Timer(name="epoch_timer", logger=None)
    batch_timer = Timer(name="batch_timer", logger=None)

    if not run_id:
        run_id = fw.generate("po", separator="-")
    logger.info(f"Registering run id {run_id} with wandb")
    wandb.init(
        # set the wandb project where this run will be logged
        project="toxicity-classifier",
        id=run_id,
        resume="allow",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": model.ARCH,
            "model": model.__class__.__name__,
            "dataset": dataloaders["train"].dataset.__class__,
            "optimizer": optimizer.__class__.__name__,
            "epochs": epochs,
        }
    )
    val_loss = 0.

    for epoch in range(epochs):
        # Training phase
        epoch_timer.start()
        model.train()
        logger.info(f"---- Starting epoch {epoch} [last={epoch_timer.last:.2f} s | val_loss={val_loss:.4f} | len={len(dataloaders['train'])}]  ----")
        for idx, sample in enumerate(dataloaders["train"]):
            batch_timer.start()
            X, y = sample[0], sample[1]
            optimizer.zero_grad()
            y_pred = model(X)
            y_oh = torch.nn.functional.one_hot(y, num_classes=2)
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
        logger.info("Validating batch")
        model.eval()
        with torch.no_grad():
            for sample in dataloaders["val"]:
                X, y = sample[0], sample[1]
                y_pred = model(X)
                y_oh = torch.nn.functional.one_hot(y, num_classes=2)
                loss = loss_fn(y_pred.squeeze(), y_oh)
                val_loss +=  loss.cpu().detach().numpy()

        val_loss /= len(dataloaders["val"])
        wandb.log({"val_loss_avg": val_loss})
        val_loss = 0.

    run_metrics(model, dataloaders["test"], device, log="wandb")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        },
        f"toxicity_classifier_{wandb.run.name}_{loss:0.4f}_{run_id}.pth"
    )
    wandb.finish()
    logger.info("DONE.")

def run_metrics(model, dataloader, device, log:str=None):
    precision = Precision()
    recall = Recall()
    def f1_score(precision, recall):
        return (2 * precision * recall) / (precision + recall + 1e-20)
    def model_transform(x):
        return torch.argmax(x.to(torch.float32), dim=2).squeeze(dim=1)
    metrics = {
        "accuracy": Accuracy(),
        "recall": recall,
        "precision": precision,
        "f1": MetricsLambda(f1_score, precision, recall),
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, model_transform=model_transform)
    logger.info(f"Running inference for {dataloader}")
    evaluator.run(dataloader)
    metrics = evaluator.state.metrics
    logger.info(f"\n{metrics}")
    if log.startswith("wandb"):
        wandb.log({
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_accuracy": metrics["accuracy"]
        })
    return metrics


def run_eval(path, classifier: Union[str, Callable]=None, device=None, run_id=None):
    device = get_device(device)
    classifier = get_classifier(classifier)

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

    # Define Model
    batch_size, _, input_dim = next(iter(dataloaders["val"]))[0].shape
    model = classifier(input_size=input_dim, hidden_size=256, output_size=2).to(device)
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Define Metrics
    def f1_score(precision, recall):
        return (2 * precision * recall) / (precision + recall + 1e-20)

    precision = Precision()
    recall = Recall()
    metrics = {
        "accuracy": Accuracy(),
        "recall": recall,
        "precision": precision,
        "confusion": ConfusionMatrix(num_classes=2),
        "f1": MetricsLambda(f1_score, precision, recall),
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    wandb.init(
        project = "toxicity-classifier",
        id = run_id,
        resume = "allow",
    )

    # Run Inference
    for dl in dataloaders:
        logger.info(f"Running inference for {dl}")
        evaluator.run(dataloaders[dl])
        metrics = evaluator.state.metrics
        logger.info(f"\n{metrics}")
        if run_id:
            confusion_table = wandb.Table(data=pd.DataFrame(metrics["confusion"]))
            metrics["confusion"] = confusion_table
            wandb.log({"metrics": metrics})
    wandb.finish()


def run_embeddings(src_csv, dest_parquet, device=None):
    device = get_device(device)
    logger.info(f"Using device={device}")

    tokenizer, embedder = build_embedding(device=device)
    # dataset_dir = Path.cwd() / "Data" / "Wikipedia-Toxic-Comments"
    df = pd.read_csv(src_csv)

    with logging_redirect_tqdm():
        embeddings = [
            embed(preprocess(text), tokenizer, embedder, device)
            for text in tqdm(df["comment_text"].values)
        ]

    logger.info(f"Storing embeddings in file {dest_parquet}")
    df["embeddings"] = [str(x) for x in embeddings]
    df.to_parquet(dest_parquet, compression='gzip')
