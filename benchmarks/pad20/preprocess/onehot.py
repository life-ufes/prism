import config
import numpy as np
import pandas as pd

from benchmarks.pad20.dataset import PAD20
from sklearn.model_selection import StratifiedGroupKFold

if __name__ == "__main__":
    print("- Loading the dataset")
    df = pd.read_csv(config.PAD_20_RAW_METADATA)

    print("- Splitting the dataset")
    # create cross-validation splits, grouping by patient and while stratifying by diagnostic
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df["folder"] = None
    for i, (_, test_indexes) in enumerate(
        kfold.split(df, df[PAD20.TARGET_COLUMN], groups=df[PAD20.PATIENT_ID])
    ):
        df.loc[test_indexes, "folder"] = i + 1

    # Validate patient id separation across folders
    patient_ids = df.groupby("folder")[PAD20.PATIENT_ID].unique()
    for i, ids in enumerate(patient_ids):
        for j, other_ids in enumerate(patient_ids):
            if i != j and set(ids).intersection(other_ids):
                raise ValueError(
                    f"Patient IDs {ids} and {other_ids} are present in the same folder {i+1} and {j+1}."
                )

    print("- Converting the labels to numbers")
    int_to_label = {0: "ACK", 1: "BCC", 2: "MEL", 3: "NEV", 4: "SCC", 5: "SEK"}
    label_to_int = {v: k for k, v in int_to_label.items()}

    df[PAD20.TARGET_COLUMN] = df[PAD20.TARGET_COLUMN].astype("category")
    df[PAD20.TARGET_NUMBER_COLUMN] = df[PAD20.TARGET_COLUMN].map(label_to_int)

    print("- Checking the target distribution")
    print(df[PAD20.TARGET_COLUMN].value_counts())
    print(f"Total number of samples: {df[PAD20.TARGET_COLUMN].value_counts().sum()}")

    # fix empty values
    df = df.replace(" ", np.nan).replace("  ", np.nan)

    # fix brazilian background
    df.loc[:, ["background_father", "background_mother"]] = df.loc[
        :, ["background_father", "background_mother"]
    ].replace("BRASIL", "BRAZIL")

    df.loc[:, PAD20.NUMERICAL_FEATURES] = (
        df.loc[:, PAD20.NUMERICAL_FEATURES].fillna(-1).astype(np.float32)
    )

    df.loc[:, PAD20.RAW_CATEGORICAL_FEATURES] = df.loc[
        :, PAD20.RAW_CATEGORICAL_FEATURES
    ].fillna("UNK")

    df = pd.get_dummies(df, columns=PAD20.RAW_CATEGORICAL_FEATURES, dtype=np.int8)

    # ensure each categorical feature has a corresponding UNK column
    for feature in PAD20.RAW_CATEGORICAL_FEATURES:
        unk_col = f"{feature}_UNK"
        if unk_col not in df.columns:
            df[unk_col] = 0

    # create one-hot-encoded metadata parent folder
    config.PAD_20_ONE_HOT_ENCODED.parent.mkdir(exist_ok=True)

    # save one-hot-encoded metadata
    df.to_csv(config.PAD_20_ONE_HOT_ENCODED)

    print("File saved to:", config.PAD_20_ONE_HOT_ENCODED)
