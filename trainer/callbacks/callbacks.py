import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from lightning.pytorch.callbacks import Callback


class CheckpointManager(Callback):
    """Handle best and last checkpoint saving."""

    def __init__(self, monitor="loss/val", mode="min", save_best=True, save_last=False):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_metric = monitor.split("/")[0]  # e.g., 'loss' from 'loss/val'
        self.stage = (
            f'_{monitor.split("/")[1]}' if "/" in monitor else "_val"
        )  # e.g., 'val' from 'loss/val'

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.save_best and not self.save_last:
            return

        metric_value = pl_module.metrics[self.stage][self.best_metric].compute().item()
        if metric_value is None:
            raise ValueError(
                f"The metric {self.monitor} is not available to check whether the model improved"
            )

        log_dir = Path(trainer.logger.log_dir)

        if self.save_last:
            torch.save(
                {"model_state_dict": pl_module.model.state_dict()},
                log_dir / "last_checkpoint.pth",
            )

        # Save best checkpoint if requested and improved
        if self.save_best:
            improved = (self.mode == "min" and metric_value < self.best_value) or (
                self.mode == "max" and metric_value > self.best_value
            )

            if improved:
                self.best_value = metric_value
                torch.save(
                    {"model_state_dict": pl_module.model.state_dict()},
                    log_dir / "best_checkpoint.pth",
                )


class TrainingHistory(Callback):
    """Save loss curves"""

    def __init__(self):
        super().__init__()
        self.metrics_history = {
            "train": {"loss": []},
            "val": {"loss": []},
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Save metrics history and create plots."""

        for stage, metric in pl_module.metrics.items():
            if "test" in stage:
                continue

            self.metrics_history[stage.replace("_", "")].get("loss").append(
                metric["loss"].compute().cpu().numpy()
            )

    def on_train_end(self, trainer, pl_module):
        log_dir = Path(trainer.logger.log_dir)

        # Restructure data for DataFrame
        history_df = pd.DataFrame(
            {
                "epoch": list(range(1, len(self.metrics_history["train"]["loss"]) + 1)),
                "train_loss": self.metrics_history["train"]["loss"],
                "val_loss": self.metrics_history["val"]["loss"],
            }
        )

        # Save to CSV
        history_df.to_csv(log_dir / "training_history.csv", index=False)
        self._plot_loss_curves(Path(trainer.logger.log_dir), self.metrics_history)

    def _plot_loss_curves(self, save_dir, metrics_history):
        """Create loss curves plot."""
        epochs = range(1, len(metrics_history["train"]["loss"]) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_history["train"]["loss"], "b-", label="Training Loss")
        plt.plot(epochs, metrics_history["val"]["loss"], "r-", label="Validation Loss")
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "loss_history.png")
        plt.close()


class TestMetricSaver(Callback):
    """Save test metrics."""

    def __init__(self, filename="test_metrics.csv"):
        super().__init__()
        self.filename = filename

    def on_test_end(self, trainer, pl_module):
        log_dir = Path(trainer.logger.log_dir)

        metrics = {}
        for metric_name, metric in pl_module.metrics["_test"].items():
            metrics[metric_name] = [metric.compute().cpu().numpy()]

        pd.DataFrame(metrics).to_csv(log_dir / self.filename, index=False)
