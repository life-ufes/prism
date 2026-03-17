import os
import torch
from torch import nn
import torch.nn.functional as F


class NaiveBayes(nn.Module):
    """Naive Bayes late-fusion head.
    This module wraps a vision model and adds a Naive Bayes component that can be fit on metadata.
    The NB component learns class-conditional means and variances for numerical features,
    and log-frequency tables for categorical features.
    During forward, it combines the NB log-probabilities with the vision model
    """

    def __init__(
        self,
        vision_model: nn.Module,
        n_classes: int,
        n_categorical_features: int,
        n_numerical_features: int,
        eps: float = 1e-6,
        laplacian_smoothing: float = 1.0,
    ):
        super().__init__()
        self.vision_model = vision_model

        # Save Hyperparameters and state flags as buffers
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32))
        self.register_buffer(
            "laplacian_smoothing",
            torch.tensor(float(laplacian_smoothing), dtype=torch.float32),
        )
        self.register_buffer("is_fit", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("n_classes", torch.tensor(n_classes, dtype=torch.long))

        # Feature index buffers
        self.register_buffer(
            "numerical_features", torch.zeros(n_numerical_features, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_features",
            torch.zeros(n_categorical_features, dtype=torch.long),
        )

        # Optional class weights
        self.register_buffer("weights", torch.zeros(n_classes, dtype=torch.float32))

        # Learned statistics (initialized empty)
        self.register_buffer(
            "means", torch.zeros(n_classes, n_numerical_features, dtype=torch.float32)
        )
        self.register_buffer(
            "variances",
            torch.zeros(n_classes, n_numerical_features, dtype=torch.float32),
        )
        self.register_buffer(
            "log_frequency_table",
            torch.zeros(n_classes, n_categorical_features, dtype=torch.float32),
        )

    @torch.no_grad()
    def fit(
        self,
        metadata,
        labels,
        categorical_features,
        numerical_features,
        n_classes,
        weights=None,
    ):
        """Fit NB statistics on provided metadata/labels.

        categorical_features and numerical_features should be lists/arrays of column indices.
        """
        # Convert to tensors on a consistent device
        if not isinstance(metadata, torch.Tensor):
            metadata = torch.as_tensor(metadata, dtype=torch.float32)

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)

        device = metadata.device

        # Store config/state in buffers
        self.n_classes = torch.tensor(int(n_classes), dtype=torch.long, device=device)

        self.numerical_features = (
            torch.as_tensor(numerical_features, dtype=torch.long, device=device)
            if len(numerical_features) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )

        self.categorical_features = (
            torch.as_tensor(categorical_features, dtype=torch.long, device=device)
            if len(categorical_features) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )

        if weights is None:
            self.weights = torch.ones(
                int(self.n_classes.item()), dtype=torch.float32, device=device
            )
        else:
            self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        n_cls = int(self.n_classes.item())
        n_num = int(self.numerical_features.numel())
        n_cat = int(self.categorical_features.numel())

        # Initialize and estimate numerical stats
        if n_num > 0:
            self.means = torch.zeros((n_cls, n_num), dtype=torch.float32, device=device)
            self.variances = torch.zeros(
                (n_cls, n_num), dtype=torch.float32, device=device
            )
            self.estimate_mean_var(metadata[:, self.numerical_features], labels)
        else:
            self.means = torch.empty(0, 0, dtype=torch.float32, device=device)
            self.variances = torch.empty(0, 0, dtype=torch.float32, device=device)

        # Initialize and estimate categorical log-frequency table
        if n_cat > 0:
            self.log_frequency_table = torch.zeros(
                (n_cls, n_cat), dtype=torch.float32, device=device
            )
            self.fill_log_frequency_table(
                metadata[:, self.categorical_features], labels
            )
        else:
            self.log_frequency_table = torch.empty(
                0, 0, dtype=torch.float32, device=device
            )

        self.is_fit = torch.tensor(True, dtype=torch.bool, device=device)

    @torch.no_grad()
    def estimate_mean_var(self, features, labels):
        for label in range(self.n_classes):
            x = features[(labels == label)]
            self.means[label, :] = torch.nanmean(x, dim=0)

            # fill nan values with mean to calculate variance:
            x_mean_fill = torch.where(
                torch.isnan(x), self.means[label, :].unsqueeze(0), x
            )
            self.variances[label, :] = (
                torch.var(x_mean_fill, dim=0, unbiased=True) + self.eps
            )

    @torch.no_grad()
    def fill_log_frequency_table(self, features: torch.Tensor, labels):
        for label in range(self.n_classes):
            self.log_frequency_table[label, :] = (
                features[(labels == label)].sum(axis=0) + self.laplacian_smoothing
            ) * self.weights[label]

        self.log_frequency_table = (
            self.log_frequency_table / self.log_frequency_table.sum(axis=0)
        )
        self.log_frequency_table = torch.log(self.log_frequency_table)

    def forward(
        self, img: torch.Tensor, meta: torch.Tensor, img_ids: tuple[str] = None
    ):
        if not bool(self.is_fit):
            raise RuntimeError("The Naive Bayes was not fit.")

        nb_log = None
        if self.numerical_features.numel() > 0:
            nb_log = self.get_numerical_features_log_probs(meta)

        if self.categorical_features.numel() > 0:
            if nb_log is not None:
                nb_log += self.get_categorical_features_log_probs(meta)
            else:
                nb_log = self.get_categorical_features_log_probs(meta)

        if self.vision_model is not None:
            logits = self.vision_model(img, meta)
            probs = F.log_softmax(logits, dim=1)
            return (probs + nb_log) if nb_log is not None else probs

        return nb_log

    def get_numerical_features_log_probs(self, batch: torch.Tensor):
        # log(normal_distribution) = -log(sqrt(2*pi*var)) -(x-u)²/2*var
        normalization_term = -torch.log(torch.sqrt(2 * torch.pi * self.variances))
        log_probs = (
            normalization_term
            - 0.5
            * (batch[:, self.numerical_features].unsqueeze(dim=1) - self.means) ** 2
            / self.variances
        )

        # sum the log probs from features that are not NaN
        # this effectively ignores missing features in the product of probabilities,
        mask = ~torch.isnan(batch[:, self.numerical_features])
        mask_expanded = mask.unsqueeze(1).expand(-1, log_probs.size(1), -1)
        return (
            torch.where(mask_expanded, log_probs, torch.zeros_like(log_probs))
            .sum(dim=2)
            .float()
        )

    def get_categorical_features_log_probs(self, batch: torch.Tensor):
        return torch.matmul(
            (batch[:, self.categorical_features] == 1).float(),
            self.log_frequency_table.t(),
        )
