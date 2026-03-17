import torch
import numpy as np
import pandas as pd
import torchmetrics
import lightning as L
import torch.nn.functional as F

from torch import nn
from torch import optim
from pathlib import Path


class Step:
    def __init__(self, ids, probs, targets, stages) -> None:
        self.ids = ids
        self.probs = probs
        self.targets = targets
        self.stages = stages


class LightningAdapter(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        n_classes,
        weights=None,
        label_smoothing=0.0,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=5,
        lr_scheduler_min_lr=5e-7,
        lr_scheduler_mode="min",
        lr_initial=5e-5,
        lr_scheduler_monitor="loss/val",
        target_number_to_label=None,
        one_cycle_lr_epochs=100,
        one_cycle_lr_steps_per_epoch=1,
        one_cycle_lr_pct_start=0.2,
        one_cycle_lr_div_factor=1000,
        one_cycle_lr_final_div_factor=1000,
        prog_bar_metrics=["loss", "recall"],
        predict_dataloader_stages=["train", "val", "test"],
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["model", "weights"])
        self.model = model
        self.weights = (
            weights.detach().clone().requires_grad_(False)
            if weights is not None
            else None
        )

        stages = ["_train", "_val", "_test"]

        metric_configs = {
            "task": "multiclass",
            "num_classes": n_classes,
        }
        self.metrics = nn.ModuleDict(
            {
                stage: nn.ModuleDict(
                    {
                        "loss": torchmetrics.MeanMetric(),
                        "acc": torchmetrics.Accuracy(**metric_configs),
                        "f1": torchmetrics.F1Score(
                            **metric_configs, threshold=0.5, average="macro"
                        ),
                        "recall": torchmetrics.Recall(
                            **metric_configs, threshold=0.5, average="macro"
                        ),
                        "auc": torchmetrics.AUROC(**metric_configs, average="macro"),
                    }
                )
                for stage in stages
            }
        )

        self.prog_bar_metrics = prog_bar_metrics
        self.prediction_steps = []

    def forward(self, *args):
        return self.model(*args)

    def _step(self, batch, batch_idx, stage):
        logits, targets, ids = self.get_logits_targets_and_ids(batch)
        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Update metrics
        probs = F.softmax(logits, dim=1)
        for name, metric in self.metrics[stage].items():
            if "loss" in name:
                metric.update(loss.detach())
            else:
                targets_for_metrics = (
                    torch.argmax(targets, dim=1)
                    if targets.dtype != torch.long
                    else targets
                )
                metric.update(probs, targets_for_metrics.long())

        return {"loss": loss, "probs": probs, "targets": targets, "ids": ids}

    def get_logits_targets_and_ids(self, batch):
        img, meta, targets, img_ids = batch
        return self.model(img, meta, img_ids), targets, img_ids

    def on_train_epoch_start(self):
        self._reset_metrics()

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "_train")

    def on_train_epoch_end(self) -> None:
        # Log all metrics
        for stage, metrics in self.metrics.items():
            if "test" in stage:
                continue
            for name, metric in metrics.items():
                self.log(
                    f"{name}/{stage.replace('_', '')}",
                    metric.compute(),
                    prog_bar=name in self.prog_bar_metrics,
                )

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "_val")

    def on_test_start(self):
        if self.trainer.logger is not None:
            self._load_best_checkpoint()

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "_test")

    def on_predict_start(self):
        if self.trainer.logger is not None:
            self._load_best_checkpoint()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits, targets, ids = self.get_logits_targets_and_ids(batch)
        probs = F.softmax(logits, dim=1)
        stages = [self.hparams.predict_dataloader_stages[dataloader_idx]] * len(ids)
        self.prediction_steps.append(Step(ids, probs, targets, stages))
        return probs, targets, ids

    def on_predict_end(self):
        probs = torch.cat([step.probs for step in self.prediction_steps]).cpu().numpy()
        targets = (
            torch.cat([step.targets for step in self.prediction_steps]).cpu().numpy()
        )
        stages = np.concatenate([step.stages for step in self.prediction_steps])
        ids = [step.ids for step in self.prediction_steps]
        is_id_tensor = isinstance(ids[0], torch.Tensor)
        ids = (
            torch.cat(ids) if isinstance(ids[0], torch.Tensor) else np.concatenate(ids)
        )

        # translate target_number back to target label
        probs = {
            self.hparams.target_number_to_label[i]: prob
            for i, prob in enumerate(probs.T)
        }
        targets = [self.hparams.target_number_to_label[l] for l in targets]

        if self.trainer.logger is not None:
            experiment_dir = Path(self.trainer.logger.log_dir)
            pd.DataFrame(
                {
                    "labels": targets,
                    "id": ids.cpu().numpy() if is_id_tensor else ids,
                    "stage": stages,
                }
                | probs
            ).to_csv(experiment_dir / "best_checkpoint_preds.csv")

        self.prediction_steps.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr_initial)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            epochs=self.hparams.one_cycle_lr_epochs,
            steps_per_epoch=self.hparams.one_cycle_lr_steps_per_epoch,
            pct_start=self.hparams.one_cycle_lr_pct_start,
            max_lr=self.hparams.lr_initial,
            div_factor=self.hparams.one_cycle_lr_div_factor,
            final_div_factor=self.hparams.one_cycle_lr_final_div_factor,
            anneal_strategy="cos",
            cycle_momentum=True,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "name": "lr",
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _load_best_checkpoint(self):
        experiment_dir = Path(self.trainer.logger.log_dir)
        ckpt = torch.load(experiment_dir / "best_checkpoint.pth")
        self.model.load_state_dict(ckpt["model_state_dict"])

    def _reset_metrics(self):
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
