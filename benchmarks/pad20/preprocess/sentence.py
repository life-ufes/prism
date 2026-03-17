import config
import numpy as np
import pandas as pd

from benchmarks.pad20.dataset import PAD20
from sklearn.model_selection import StratifiedGroupKFold

from utils.names import FIELD_TO_LABEL

def generate_sentence(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["age"].apply(lambda x: int(x) if pd.notna(x) else "unknown")
    df["fitspatrick"] = df["fitspatrick"].apply(
        lambda x: int(x) if pd.notna(x) else "unknown"
    )
    df["diameter_1"] = df["diameter_1"].apply(
        lambda x: int(x) if pd.notna(x) else "unknown"
    )
    df["diameter_2"] = df["diameter_2"].apply(
        lambda x: int(x) if pd.notna(x) else "unknown"
    )
    df = df.reset_index(drop=True)
    for col in df.select_dtypes(include="category").columns:
        if "unknown" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories("unknown")

    df = df.fillna("unknown")

    features = [
        "age",
        "gender",
        "background_mother",
        "background_father",
        "has_piped_water",
        "has_sewage_system",
        "smoke",
        "drink",
        "pesticide",
        "fitspatrick",
        "skin_cancer_history",
        "cancer_history",
        "region",
        "grew",
        "itch",
        "bleed",
        "hurt",
        "changed",
        "elevation",
        "diameter_1",
        "diameter_2",
    ]

    sentences = []
    for _, row in df.iterrows():
        anamnese = "Patient History: "
        for col in features:
            anamnese += f"{FIELD_TO_LABEL[col]}: {str(row[col]).lower()}, "
        anamnese = anamnese[:-2] + "."
        sentences.append(anamnese)
    df["sentence"] = pd.Series(sentences)

    return df[
        [
            PAD20.PATIENT_ID,
            PAD20.TARGET_COLUMN,
            PAD20.TARGET_NUMBER_COLUMN,
            PAD20.IMAGE_COLUMN,
            "folder",
            "sentence",
        ]
    ]


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

    df = generate_sentence(df)

    config.PAD_20_SENTENCE.parent.mkdir(exist_ok=True)
    df.to_csv(config.PAD_20_SENTENCE)

    print("File saved to:", config.PAD_20_SENTENCE)
