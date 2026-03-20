import torch
import config
import pandas as pd
import lightning as L

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from models.factory import ClassifierFactory

from trainer.callbacks.callbacks import (
    CheckpointManager,
    TestMetricSaver,
    TrainingHistory,
)
from trainer.lightning import LightningAdapter

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from benchmarks.milk10k.dataset import MILK10K, MILK10KSentenceEmbedding
from benchmarks.milk10k.augmentation import ImgTrainTransform, ImgEvalTransform

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("MILK10K")


@ex.config
def cfg():
    _model_name = "mobilenet-v3"
    _batch_size = 32
    _epochs = 100
    _n_classes = 11
    _folder = 1
    _comb_method = None
    _preprocessing = "one_hot"
    _early_stop_metric = "loss/val"
    _early_stop_patience = 10
    _weight_by_frequency = True
    _lr_initial = 1e-4
    _lr_scheduler_factor = 0.1
    _lr_scheduler_patience = 5
    _lr_scheduler_min_lr = 1e-6
    _n_workers_train_dataloader = 8
    _n_workers_val_dataloader = 4
    _initial_checkpoint = None
    _checkpoint_backbone = None  # used for the Naive Bayes
    _results_dir = (
        Path(f"benchmarks")
        / "milk10k"
        / "results"
        / (_comb_method if _comb_method is not None else "no_metadata")
    )
    _version = datetime.now().strftime("%Y%m%d_%H%M%S")
    _experiment_path = _results_dir / _model_name / _version

    _metadata_columns = MILK10K.METADATA_COLUMNS

    _naive_bayes_categorical_features = [
        "sex_female",
        "sex_male",
        "skin_tone_class_1",
        "skin_tone_class_2",
        "skin_tone_class_3",
        "skin_tone_class_4",
        "skin_tone_class_5",
        "site_foot",
        "site_genital",
        "site_hand",
        "site_head_neck_face",
        "site_lower_extremity",
        "site_trunk",
        "site_upper_extremity",
    ]

    _naive_bayes_numerical_features = [
        "age_approx",
    ]


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
    _checkpoint_backbone,
    _preprocessing,
):

    # define a device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_sentence_embedding = _preprocessing != "one_hot"
    _n_metadata = len(_metadata_columns) if not is_sentence_embedding else 768

    # read one hot encoded metadata
    if _checkpoint_backbone is not None:
        # use the same metadata as the backbone checkpoint to avoid data leakage
        # (the metadata used to train the backbone is saved in its results folder)
        csv_path = _checkpoint_backbone.parent / "metadata.csv"
    else:
        if not is_sentence_embedding:
            csv_path = config.MILK10K_TRAIN_ONE_HOT_ENCODED
        else:
            csv_path = config.MILK10K_TRAIN_SENTENCE

    # split train, val
    df = pd.read_csv(csv_path, index_col=0)
    val_mask = df["folder"] == _folder
    train, val = df[~val_mask], df[val_mask]

    print("Train set:")
    print(train[MILK10K.TARGET_COLUMN].value_counts())

    # save metadata into results folder
    df.to_csv(_experiment_path / "metadata.csv")

    # load the datasets
    if is_sentence_embedding:
        sentence_model = SentenceTransformer(
            "sentence-transformers/paraphrase-albert-small-v2"
        )
        train_dataset = MILK10KSentenceEmbedding(
            train,
            sentence_model,
            clinical_transforms=ImgTrainTransform(),
            dermoscopic_transforms=ImgTrainTransform(),
        )
        val_dataset = MILK10KSentenceEmbedding(
            val,
            sentence_model,
            clinical_transforms=ImgTrainTransform(),
            dermoscopic_transforms=ImgTrainTransform(),
        )
    else:
        train_dataset = MILK10K(
            train,
            clinical_transforms=ImgTrainTransform(),
            dermoscopic_transforms=ImgTrainTransform(),
            meta_ordered_features=_metadata_columns,
        )
        val_dataset = MILK10K(
            val,
            clinical_transforms=ImgEvalTransform(),
            dermoscopic_transforms=ImgEvalTransform(),
            meta_ordered_features=_metadata_columns,
        )

    # define weights inversely proportional to class frequency
    weights = (
        len(train_dataset) / torch.bincount(train_dataset.targets).to(device)
        if _weight_by_frequency
        else None
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_batch_size,
        num_workers=_n_workers_train_dataloader,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_batch_size,
        num_workers=_n_workers_val_dataloader,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    model = ClassifierFactory.get(
        _n_classes,
        _model_name,
        comb_method=_comb_method,
        n_metadata=_n_metadata,
        vision_checkpoint=_checkpoint_backbone,
        n_categorical_metadata=len(_naive_bayes_categorical_features) if _comb_method == "naive_bayes" else None,
        n_numerical_metadata=len(_naive_bayes_numerical_features) if _comb_method == "naive_bayes" else None,
    )

    if _comb_method == "naive_bayes":
        train_meta = train[_metadata_columns]
        model.fit(
            torch.Tensor(train_meta.values),
            labels=train[MILK10K.TARGET_NUMBER_COLUMN].values,
            categorical_features_indexes=[
                list(train_meta.columns).index(x)
                for x in _naive_bayes_categorical_features
            ],
            numerical_features_indexes=[
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

    # When evaluating the backbone with a Naive Bayes, we don't want to train it,
    # so we set max_epochs to 0 to skip training and only evaluate the backbone on the validation set
    max_epochs = 0 if _comb_method == 'naive_bayes' else _epochs

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
