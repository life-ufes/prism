import config
import numpy as np
import pandas as pd
from benchmarks.milk10k.dataset import MILK10K
from sklearn.model_selection import StratifiedKFold


def _pivot_dataframe(df):
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


def _preprocess(features_path, labels_path, images_folder, output_path):
    df = pd.read_csv(features_path, index_col=0)
    df = _pivot_dataframe(df)

    labels = pd.read_csv(labels_path, index_col=0) if labels_path is not None else None

    df[MILK10K.TARGET_COLUMN] = labels.idxmax(axis=1)

    int_to_label = {
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

    label_to_int = {v: k for k, v in int_to_label.items()}

    df.loc[:, MILK10K.TARGET_NUMBER_COLUMN] = df[MILK10K.TARGET_COLUMN].map(
        label_to_int
    )

    df = df.reset_index()
    df = df.rename(columns={"index": MILK10K.LESION_ID_COLUMN})

    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )  # The MILK10k dataset has no patient-level information.

    df["folder"] = None
    for i, (_, test_indexes) in enumerate(kfold.split(df, df[MILK10K.TARGET_COLUMN])):
        df.loc[test_indexes, "folder"] = i + 1

    print("- Checking the target distribution")
    print(df[MILK10K.TARGET_COLUMN].value_counts())
    print(f"Total number of samples: {df[MILK10K.TARGET_COLUMN].value_counts().sum()}")

    for image_col in [
        MILK10K.CLINICAL_IMAGE_COLUMN,
        MILK10K.DERMATOSCOPIC_IMAGE_COLUMN,
    ]:
        df.loc[:, image_col] = df.apply(
            lambda row: images_folder
            / row[MILK10K.LESION_ID_COLUMN]
            / f"{row[image_col]}.jpg",
            axis=1,
        )

    df[MILK10K.NUMERICAL_FEATURES] = (
        df[MILK10K.NUMERICAL_FEATURES].fillna(-1).astype(np.float32)
    )

    df = pd.get_dummies(df, columns=MILK10K.RAW_CATEGORICAL_FEATURES, dtype=np.int8)

    output_path.parent.mkdir(exist_ok=True)

    df.to_csv(output_path)

    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    for features_path, labels_path, images_folder, output_path in [
        (
            config.MILK10K_TRAIN_RAW_METADATA,
            config.MILK10K_TRAIN_LABELS,
            config.MILK10K_TRAIN_IMAGES_FOLDER,
            config.MILK10K_TRAIN_ONE_HOT_ENCODED,
        ),
    ]:
        _preprocess(features_path, labels_path, images_folder, output_path)
