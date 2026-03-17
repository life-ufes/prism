from benchmarks.milk10k.bayesian.dataset import MILK10KBayesian
from benchmarks.milk10k.bayesian.model import Milk10kBayesianNetwork
from benchmarks.milk10k.dataset import MILK10K
from benchmarks.trainpyro import train
from sacred import Experiment
from torchmetrics.classification import Accuracy
from sacred.observers import FileStorageObserver
from config import MILK10K_BAYESIAN_DATA
from functools import partial
from pathlib import Path
import os
import time
import pandas as pd
from torch.utils.data import DataLoader

NUM_CLASSES = len(MILK10K.LABELS)
BACC = Accuracy(task="multiclass", num_classes=NUM_CLASSES, average="macro").cuda()


def get_dataloader(model, filename, batch_size, features, stage):
    if stage not in ["train", "val"]:
        raise ValueError("The split should be one of: train, val.")
    metadata = pd.read_csv(MILK10K_BAYESIAN_DATA / model / filename)
    dataset = MILK10KBayesian(metadata, stage=stage, features=features)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=stage == "train",
        pin_memory=True,
        num_workers=16,
    )


ex = Experiment("milk10k_bn")


@ex.config
def cnfg():
    features = MILK10KBayesian.DEFAULT_FEATURES
    folder = 1
    epochs = 100
    batch_size = 65
    learning_rate = 2.5e-3
    early_stop_patience = 10
    early_stop_metric = "bacc"
    model_name = "mobilenet-v3"
    save_folder = Path(
        f"benchmarks/milk10k/results/bayesiannetwork/{str(time.time()).replace('.', '')}/{model_name}/folder_{str(folder)}"
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
    save_folder,
):

    filename = f"folder_{folder}.csv"
    df = pd.read_csv(MILK10K_BAYESIAN_DATA / model_name / filename)
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(save_folder / "metadata.csv", index=False)

    train(
        Milk10kBayesianNetwork(),
        partial(get_dataloader, model=model_name, filename=filename),
        features=features,
        folder=folder,
        save_folder=save_folder,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stop_patience=early_stop_patience,
        early_stop_metric=early_stop_metric,
        n_classes=NUM_CLASSES,
    )
