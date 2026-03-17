import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PAD20(Dataset):

    FEATURES = [
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

    def __init__(
        self, metadata: pd.DataFrame, image_folder=None, stage="train", features=None
    ):
        super().__init__()
        self.metadata = metadata
        self.metadata = self.metadata[self.metadata["stage"] == stage]
        all_features = PAD20.FEATURES if features is None else features

        self.int_to_label = {0: "ACK", 1: "BCC", 2: "MEL", 3: "NEV", 4: "SCC", 5: "SEK"}

        if features:
            for ft in features:
                if ft not in all_features:
                    raise ValueError(f"Invalid feature: {ft}")
            self.features = features
        else:
            self.features = all_features

    def to_label(self, diagnostic_number):
        return self.int_to_label[diagnostic_number]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        return (
            row["img_id"],
            torch.tensor(row.loc[self.features].to_numpy(dtype=np.float32)),
            torch.tensor(row.loc["diagnostic_number"], dtype=torch.long),
        )
