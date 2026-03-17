import os
import config
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from benchmarks.pad20.dataset import PAD20
from sklearn.preprocessing import LabelEncoder


def label_encode_non_nans(df, label_encoder, feature):
    was_na = df[feature].isna()
    df[feature] = label_encoder.fit_transform(df[feature])
    df.loc[was_na, feature] = np.nan


def preprocess(df_folder, folder_path, label_encoder):
    cnn_results = pd.read_csv(folder_path / "best_checkpoint_preds.csv")

    df_folder["img_id"] = df_folder["img_id"].str.replace(".png", "")

    df_folder = df_folder.merge(
        cnn_results[["id", "stage"] + PAD20.LABELS],
        how="left",
        left_on="img_id",
        right_on="id",
    )

    df_folder = df_folder.rename(
        columns={f"diagnostic_cnn_{l}": l for l in PAD20.LABELS}
    )

    # encode targets
    df_folder["diagnostic_number"] = label_encoder.fit_transform(
        df_folder["diagnostic"]
    )

    # discretize numerical features
    df_folder["age_group"] = pd.cut(
        df_folder["age"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    df_folder["diameter"] = pd.cut(
        df_folder["diameter"],
        bins=[0, 5, 10, 15, 20, 25, 30, 35, 120],
        include_lowest=True,
    )

    # encode categorical features as numbers
    for feature in ["age_group", "diameter", "region"]:
        label_encode_non_nans(df_folder, label_encoder, feature)

    # save csv
    return df_folder[
        [
            "img_id",
            "stage",
            "diagnostic",
            "diagnostic_number",
            "itch",
            "grew",
            "hurt",
            "changed",
            "bleed",
            "elevation",
            "age_group",
            "diameter",
            "region",
        ]
        + PAD20.LABELS
    ]


def save_csv(df, missing, raw, save_folder, folder_number):
    file_name = f"folder_{folder_number}_raw" if raw else f"folder_{folder_number}"
    file_name += f"_missing_{missing}.csv"
    df.to_csv(save_folder / file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PAD20 dataset for Bayesian Network"
    )
    parser.add_argument(
        "--no-metadata-timestamp",
        type=str,
        help="Timestamp folder of the CNN-only experiment (no metadata)",
    )
    args = parser.parse_args()
    try:
        cnn_only_results_folder = (
            next(Path(".").rglob(str(args.no_metadata_timestamp))) / "no_metadata"
        )
    except StopIteration:
        raise FileNotFoundError(
            f"No folder found for experiment {args.no_metadata_timestamp}. \
                                You should first run the kfold.py on the PAD20 benchmark to generate CNN results without using metadata."
        )

    df = pd.read_csv(config.PAD_20_RAW_METADATA)
    df = df.replace("UNK", np.nan)

    # group diameter as max(diameter_1, diameter_2)
    df["diameter"] = df.apply(
        lambda row: max(row["diameter_1"], row["diameter_2"]), axis=1
    )

    # fix booleans
    df = df.replace(["True", "False"], [1, 0])

    label_encoder = LabelEncoder()

    save_folder = config.PAD_20_ONE_HOT_ENCODED.parent / "bayesian"
    os.makedirs(save_folder, exist_ok=True)

    for model_path in (p for p in cnn_only_results_folder.iterdir() if p.is_dir()):
        model_name = model_path.stem
        for folder_number in range(1, 6):
            folder_path = model_path / f"folder_{folder_number}"
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder {folder_path} does not exist.")

            df_folder = df.copy()
            os.makedirs(save_folder / model_name, exist_ok=True)
            df_folder = preprocess(df_folder, folder_path, label_encoder)
            df_folder.to_csv(save_folder / model_name / f"folder_{folder_number}.csv")

    print(f"Preprocessed data saved to {save_folder}")
