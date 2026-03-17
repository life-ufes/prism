import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from benchmarks.milk10k.dataset import MILK10K
import config


def _pivot_dataframe(df):
    print(df.head())
    pivoted = df.pivot_table(
        index=MILK10K.LESION_ID_COLUMN,
        columns=MILK10K.IMAGE_TYPE_COLUMN,
        values="isic_id",
        aggfunc="first",
    ).reset_index()
    meta = df.groupby(MILK10K.LESION_ID_COLUMN).first().reset_index()
    meta.columns = meta.columns.str.replace("MONET_", "clinical_")
    pivoted = pivoted.merge(meta, on=MILK10K.LESION_ID_COLUMN)

    train_derm = df[df[MILK10K.IMAGE_TYPE_COLUMN].str.contains("dermos")]
    train_derm.columns = train_derm.columns.str.replace("MONET_", "dermoscopic_")

    pivoted: pd.DataFrame = pivoted.merge(
        train_derm[[c for c in train_derm.columns if c.startswith("dermoscopic_")]],
        on=[MILK10K.LESION_ID_COLUMN],
        how="left",
    )
    pivoted.rename(
        columns={
            "clinical: close-up": "image_clinical",
            "dermoscopic": "image_dermoscopic",
        },
        inplace=True,
    )
    pivoted.drop(
        columns=[
            MILK10K.IMAGE_TYPE_COLUMN,
            "attribution",
            "image_manipulation",
            "copyright_license",
            "isic_id",
        ],
        inplace=True,
    )
    pivoted = pivoted.set_index(MILK10K.LESION_ID_COLUMN)
    return pivoted


def label_encode_non_nans(df, feature):
    was_na = df[feature].isna()
    df[feature] = LabelEncoder().fit_transform(df[feature])
    df.loc[was_na, feature] = np.nan


def preprocess(df_folder, folder_path):
    cnn_results = pd.read_csv(folder_path / "best_checkpoint_preds.csv")

    df_folder = df_folder.merge(
        cnn_results[["id", "stage"] + MILK10K.LABELS],
        how="left",
        left_on="lesion_id",
        right_on="id",
    )

    df_folder = df_folder.rename(
        columns={f"diagnostic_cnn_{l}": l for l in MILK10K.LABELS}
    )

    # encode targets
    df_folder[MILK10K.TARGET_NUMBER_COLUMN] = LabelEncoder().fit_transform(
        df_folder[MILK10K.TARGET_COLUMN]
    )

    # discretize numerical features
    df_folder["age_group"] = pd.cut(
        df_folder["age_approx"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    )

    # encode categorical features as numbers
    for feature in ["age_group", "site", "sex"]:
        label_encode_non_nans(df_folder, feature)

    # save csv
    return df_folder[
        [
            MILK10K.DERMATOSCOPIC_IMAGE_COLUMN,
            "stage",
            MILK10K.TARGET_COLUMN,
            MILK10K.TARGET_NUMBER_COLUMN,
            "skin_tone_class",
            "sex",
            "age_group",
            "site",
        ]
        + MILK10K.LABELS
    ]


def save_csv(df, missing, raw, save_folder, folder_number):
    file_name = f"folder_{folder_number}_raw" if raw else f"folder_{folder_number}"
    file_name += f"_missing_{missing}.csv"
    df.to_csv(save_folder / file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MILK10K dataset for Bayesian Network"
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
                                You should first run the kfold.py on the MILK10K benchmark to generate CNN results without using metadata."
        )

    df = pd.read_csv(config.MILK10K_TRAIN_RAW_METADATA, index_col=0)
    df = _pivot_dataframe(df)

    labels = pd.read_csv(config.MILK10K_TRAIN_LABELS, index_col=0)

    df[MILK10K.TARGET_COLUMN] = labels.idxmax(axis=1)

    df.loc[:, MILK10K.TARGET_NUMBER_COLUMN] = LabelEncoder().fit_transform(
        df[MILK10K.TARGET_COLUMN]
    )

    save_folder = config.MILK10K_TRAIN_ONE_HOT_ENCODED.parent / "bayesian"
    os.makedirs(save_folder, exist_ok=True)

    for model_path in (p for p in cnn_only_results_folder.iterdir() if p.is_dir()):
        model_name = model_path.stem
        for folder_number in range(1, 6):
            folder_path = model_path / f"folder_{folder_number}"
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder {folder_path} does not exist.")

            df_folder = df.copy()
            os.makedirs(save_folder / model_name, exist_ok=True)
            df_folder = preprocess(
                df_folder,
                folder_path,
            )
            df_folder.to_csv(save_folder / model_name / f"folder_{folder_number}.csv")

    print(f"Preprocessed data saved to {save_folder}")
