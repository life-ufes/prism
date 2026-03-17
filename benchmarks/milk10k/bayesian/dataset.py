import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from benchmarks.milk10k.dataset import MILK10K


class MILK10KBayesian(Dataset):
    """Dataset that serves MILK10K metadata to the Bayesian network."""

    CATEGORICAL_FEATURES = ["sex", "skin_tone_class", "site", "age_group"]
    CNN_PROB_FEATURES = MILK10K.LABELS
    DEFAULT_FEATURES = CATEGORICAL_FEATURES + CNN_PROB_FEATURES

    AGE_GROUP_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    AGE_GROUP_CARDINALITY = len(AGE_GROUP_BINS) - 1

    SITE_LEVELS = [
        "foot",
        "genital",
        "hand",
        "head_neck_face",
        "lower_extremity",
        "trunk",
        "upper_extremity",
    ]
    SKIN_TONE_LEVELS = [0, 1, 2, 3, 4, 5]
    SEX_LEVELS = ["female", "male"]

    def __init__(
        self, metadata: pd.DataFrame, stage: str = "train", features: list | None = None
    ):
        super().__init__()
        if "stage" not in metadata.columns:
            raise ValueError(
                "Metadata must contain a 'stage' column produced by the MILK10K Bayesian preprocessor."
            )

        if stage not in metadata["stage"].unique():
            raise ValueError(
                f"Stage '{stage}' not found in metadata. Available stages: {sorted(metadata['stage'].unique())}"
            )

        self.metadata = (
            metadata[metadata["stage"] == stage].copy().reset_index(drop=True)
        )

        if self.metadata.empty:
            raise ValueError(f"No samples available for stage {stage}.")

        self.features = features if features is not None else self.DEFAULT_FEATURES
        missing_features = set(self.features) - set(self.metadata.columns)
        if missing_features:
            raise ValueError(
                f"Missing features in metadata: {sorted(missing_features)}"
            )

        self.int_to_label = {
            0: "AKIEC",
            1: "BCC",
            2: "BEN_OTH",
            3: "BKL",
            4: "DF",
            5: "INF",
            6: "MAL_OTH",
            7: "MEL",
            8: "NV",
            9: "SCCKA",
            10: "VASC",
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        img_id = row[MILK10K.DERMATOSCOPIC_IMAGE_COLUMN]
        features = torch.tensor(row.loc[self.features].to_numpy(dtype=np.float32))
        label = torch.tensor(row[MILK10K.TARGET_NUMBER_COLUMN], dtype=torch.long)
        return img_id, features, label

    def to_label(self, diagnostic_number: int):
        return self.int_to_label.get(int(diagnostic_number))
