from benchmarks.pad20.bayesian.dataset import PAD20
from torch.utils.data import DataLoader
from benchmarks.trainpyro import train
from sacred import Experiment
from torchmetrics.classification import Accuracy
from sacred.observers import FileStorageObserver
from config import PAD_20_BAYESIAN_DATA
from functools import partial
from pathlib import Path
import time
import pandas as pd
from benchmarks.pad20.bayesian.model import HeMaskedBayesianNetwork

BACC = Accuracy(task="multiclass", num_classes=6, average="macro").cuda()


def get_dataloader(model, filename, batch_size, features, stage):
    if stage not in ["train", "val", "test"]:
        raise ValueError("The split should be one of: train, val, test.")
    metadata = pd.read_csv(PAD_20_BAYESIAN_DATA / model / filename, index_col=0)
    return DataLoader(
        PAD20(metadata, stage=stage, features=features),
        batch_size=batch_size,
        shuffle=stage == "train",
        pin_memory=True,
        num_workers=16,
    )


ex = Experiment("pad20_bn")


@ex.config
def cnfg():
    features = [
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
        "region",
        "diameter",
        "age_group",
        "ACK",
        "BCC",
        "NEV",
        "MEL",
        "SEK",
        "SCC",
    ]

    folder = 1
    epochs = 100
    batch_size = 65
    learning_rate = 2.5e-3
    early_stop_patience = 10
    early_stop_metric = "bacc"
    model_name = "mobilenet-v3"
    append_observer = True
    save_folder = Path(
        f"benchmarks/pad20/results/bayesiannetwork/{str(time.time()).replace('.', '')}/{model_name}/folder_{str(folder)}"
    )


@ex.automain
def main(
    features,
    epochs,
    batch_size,
    learning_rate,
    early_stop_patience,
    early_stop_metric,
    model_name,
    folder,
    append_observer,
    save_folder,
):

    filename = f"folder_{folder}.csv"
    # save metadata
    df = pd.read_csv(PAD_20_BAYESIAN_DATA / model_name / filename, index_col=0)
    df.to_csv(save_folder / "metadata.csv", index=False)

    train(
        HeMaskedBayesianNetwork(),
        partial(get_dataloader, model=model_name, filename=filename),
        features=features,
        folder=folder,
        save_folder=save_folder,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stop_patience=early_stop_patience,
        early_stop_metric=early_stop_metric,
    )
