from sentence_transformers import SentenceTransformer
import torch
import config
import pandas as pd
import lightning as L

from pathlib import Path

from benchmarks.pad20.dataset import PAD20, PAD20SentenceEmbedding
from datetime import datetime
from torch.utils.data import DataLoader

from models.factory import ClassifierFactory
from trainer.lightning import LightningAdapter
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from sacred import Experiment

from benchmarks.pad20.augmentation import ImgTrainTransform, ImgEvalTransform
from trainer.callbacks.callbacks import (
    CheckpointManager,
    TrainingHistory,
    TestMetricSaver,
)

ex = Experiment("PAD20")


@ex.config
def cfg():
    _model_name = "mobilenet-v3"
    _batch_size = 65
    _epochs = 100
    _n_classes = 6
    _folder = 1
    _comb_method = "naive_bayes"
    _preprocessing = "one_hot"
    _early_stop_metric = "loss/val"
    _early_stop_patience = 10
    _weight_by_frequency = True
    _lr_initial = 1e-5
    _lr_scheduler_factor = 0.1
    _lr_scheduler_patience = 5
    _lr_scheduler_min_lr = 1e-7
    _n_workers_train_dataloader = 8
    _n_workers_val_dataloader = 4
    _checkpoint_backbone = None
    _results_dir = (
        Path(f"benchmarks")
        / "pad20"
        / "results"
        / (_comb_method if _comb_method is not None else "no_metadata")
    )
    _version = datetime.now().strftime("%Y%m%d_%H%M%S")
    _experiment_path = _results_dir / _model_name / _version

    _metadata_columns = PAD20.METADATA_COLUMNS

    _naive_bayes_categorical_features = [
        "skin_cancer_history_False",
        "skin_cancer_history_True",
        "cancer_history_False",
        "cancer_history_True",
        "region_ABDOMEN",
        "region_ARM",
        "region_BACK",
        "region_CHEST",
        "region_EAR",
        "region_FACE",
        "region_FOOT",
        "region_FOREARM",
        "region_HAND",
        "region_LIP",
        "region_NECK",
        "region_NOSE",
        "region_SCALP",
        "region_THIGH",
        "itch_False",
        "itch_True",
        "grew_False",
        "grew_True",
        "hurt_False",
        "hurt_True",
        "changed_False",
        "changed_True",
        "bleed_False",
        "bleed_True",
        "elevation_False",
        "elevation_True",
    ]

    _naive_bayes_numerical_features = ["age"]


@ex.automain
def main(
    _batch_size,
    _epochs,
    _model_name,
    _n_classes,
    _early_stop_metric,
    _early_stop_patience,
    _weight_by_frequency,
    _folder,
    _lr_initial,
    _lr_scheduler_factor,
    _lr_scheduler_patience,
    _lr_scheduler_min_lr,
    _results_dir,
    _version,
    _comb_method,
    _n_workers_train_dataloader,
    _n_workers_val_dataloader,
    _experiment_path,
    _metadata_columns,
    _naive_bayes_categorical_features,
    _naive_bayes_numerical_features,
    _preprocessing,
    _checkpoint_backbone,
):

    is_sentence_embedding = _preprocessing != "one_hot"
    _n_metadata = len(_metadata_columns) if not is_sentence_embedding else 768

    # load the same metadata as used during pretraining or one-hot encoded metadata
    if _checkpoint_backbone:
        csv_path = _checkpoint_backbone.parent / "metadata.csv"
    else:
        csv_path = (
            config.PAD_20_SENTENCE
            if is_sentence_embedding
            else config.PAD_20_ONE_HOT_ENCODED
        )

    df = pd.read_csv(csv_path, index_col=0)

    # save metadata into results folder
    df.to_csv(_experiment_path / "metadata.csv")

    # split train, val
    val_mask = df["folder"] == _folder
    train, val = df[~val_mask], df[val_mask]

    # load the datasets
    if is_sentence_embedding:
        sentence_model = SentenceTransformer(
            "sentence-transformers/paraphrase-albert-small-v2"
        )
        train_dataset = PAD20SentenceEmbedding(
            train, sentence_model, transforms=ImgTrainTransform()
        )
        val_dataset = PAD20SentenceEmbedding(
            val, sentence_model, transforms=ImgEvalTransform()
        )
    else:
        train_dataset = PAD20(
            train,
            transforms=ImgTrainTransform(),
            meta_ordered_features=_metadata_columns,
        )
        val_dataset = PAD20(
            val, transforms=ImgEvalTransform(), meta_ordered_features=_metadata_columns
        )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_batch_size,
        num_workers=_n_workers_train_dataloader,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=_batch_size, num_workers=_n_workers_val_dataloader
    )

    # define a device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define weights inversely proportional to class frequency
    weights = (
        len(train_dataset) / torch.bincount(train_dataset.targets).to(device)
        if _weight_by_frequency
        else None
    )

    # define a model
    model = ClassifierFactory.get(
        _n_classes,
        _model_name,
        comb_method=_comb_method,
        n_metadata=_n_metadata,
        vision_checkpoint=_checkpoint_backbone,
        n_categorical_metadata=(
            len(_naive_bayes_categorical_features)
            if _comb_method == "naive_bayes"
            else None
        ),
        n_numerical_metadata=(
            len(_naive_bayes_numerical_features)
            if _comb_method == "naive_bayes"
            else None
        ),
    )

    if _comb_method == "naive_bayes":
        train_meta = train[_metadata_columns]
        model.fit(
            torch.Tensor(train_meta.values),
            labels=train[PAD20.TARGET_NUMBER_COLUMN].values,
            categorical_features=[
                list(train_meta.columns).index(x)
                for x in _naive_bayes_categorical_features
            ],
            numerical_features=[
                list(train_meta.columns).index(x)
                for x in _naive_bayes_numerical_features
            ],
            weights=weights,
            n_classes=_n_classes,
        )

    # transform nn.Module into L.LightningModule
    model = LightningAdapter(
        model,
        n_classes=_n_classes,
        weights=weights,
        target_number_to_label=train_dataset.get_target_number_to_label(),
        lr_initial=_lr_initial,
        lr_scheduler_patience=_lr_scheduler_patience,
        lr_scheduler_factor=_lr_scheduler_factor,
        lr_scheduler_min_lr=_lr_scheduler_min_lr,
        lr_scheduler_monitor=_early_stop_metric,
    ).to(device)

    # define a prettier progress bar
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter=" , ",
            metrics_format=".3f",
        ),
        leave=True,
    )

    # define an early stopping callback
    early_stopping = EarlyStopping(
        _early_stop_metric,
        mode="min" if "loss" in _early_stop_metric else "max",
        patience=_early_stop_patience,
        verbose=True,
    )

    # define a callback to log the learning rate to tensorboard
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    # define a callback to save the best and last checkpoints
    checkpoint_callback = CheckpointManager(
        monitor=_early_stop_metric,
        mode="min" if "loss" in _early_stop_metric else "max",
        save_best=True,
        save_last=False,
    )

    # define a callback plot training curves.
    summary = TrainingHistory()

    # define a callback to save the final metrics on test set
    metrics_saver = TestMetricSaver()

    # for the naive bayes, we just load the backbone and do not further train it
    max_epochs = 0 if _checkpoint_backbone is not None else _epochs

    # define a trainer with early stopping based on loss and save the results
    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        enable_checkpointing=False,
        logger=TensorBoardLogger(
            save_dir=_results_dir, name=_model_name, version=_version
        ),
        # limit_predict_batches=1, limit_test_batches=1, limit_train_batches=1, limit_val_batches=1,
        callbacks=[
            lr_callback,
            progress_bar,
            early_stopping,
            checkpoint_callback,
            summary,
            metrics_saver,
        ],
    )

    # fit trainer
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # eval best checkpoint on val/test partition and save metrics
    trainer.test(model, dataloaders=val_dataloader)

    # create a new train_dataloader without shuffling for prediction
    train_dataloader = DataLoader(
        train_dataset, batch_size=_batch_size, num_workers=_n_workers_train_dataloader
    )
    trainer.predict(model, dataloaders=[train_dataloader, val_dataloader])
