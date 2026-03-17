import torch
import config
import pandas as pd
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image


class PAD20(Dataset):
    IMAGE_COLUMN = "img_id"
    PATIENT_ID = "patient_id"
    LESION_ID = "lesion_id"
    LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    TARGET_COLUMN = "diagnostic"
    TARGET_NUMBER_COLUMN = "diagnostic_number"

    RAW_CATEGORICAL_FEATURES = [
        "smoke",
        "drink",
        "background_father",
        "background_mother",
        "pesticide",
        "gender",
        "skin_cancer_history",
        "cancer_history",
        "has_piped_water",
        "has_sewage_system",
        "fitspatrick",
        "region",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
    ]

    NUMERICAL_FEATURES = ["age", "diameter_1", "diameter_2"]
    CNN_FEATURES = [
        f"cnn_prob_{lesion}" for lesion in ["ACK", "BCC", "MEL", "SCC", "SEK", "NEV"]
    ]
    CATEGORICAL_FEATURES = [
        "smoke_False",
        "smoke_True",
        "drink_False",
        "drink_True",
        "background_father_POMERANIA",
        "background_father_GERMANY",
        "background_father_BRAZIL",
        "background_father_NETHERLANDS",
        "background_father_ITALY",
        "background_father_POLAND",
        "background_father_UNK",
        "background_father_PORTUGAL",
        "background_father_CZECH",
        "background_father_AUSTRIA",
        "background_father_SPAIN",
        "background_father_ISRAEL",
        "background_mother_POMERANIA",
        "background_mother_ITALY",
        "background_mother_GERMANY",
        "background_mother_BRAZIL",
        "background_mother_UNK",
        "background_mother_POLAND",
        "background_mother_NORWAY",
        "background_mother_PORTUGAL",
        "background_mother_NETHERLANDS",
        "background_mother_FRANCE",
        "background_mother_SPAIN",
        "pesticide_False",
        "pesticide_True",
        "gender_FEMALE",
        "gender_MALE",
        "skin_cancer_history_True",
        "skin_cancer_history_False",
        "cancer_history_True",
        "cancer_history_False",
        "has_piped_water_True",
        "has_piped_water_False",
        "has_sewage_system_True",
        "has_sewage_system_False",
        "fitspatrick_3.0",
        "fitspatrick_1.0",
        "fitspatrick_2.0",
        "fitspatrick_4.0",
        "fitspatrick_5.0",
        "fitspatrick_6.0",
        "region_ARM",
        "region_NECK",
        "region_FACE",
        "region_HAND",
        "region_FOREARM",
        "region_CHEST",
        "region_NOSE",
        "region_THIGH",
        "region_SCALP",
        "region_EAR",
        "region_BACK",
        "region_FOOT",
        "region_ABDOMEN",
        "region_LIP",
        "itch_False",
        "itch_True",
        "itch_UNK",
        "grew_False",
        "grew_True",
        "grew_UNK",
        "hurt_False",
        "hurt_True",
        "hurt_UNK",
        "changed_False",
        "changed_True",
        "changed_UNK",
        "bleed_False",
        "bleed_True",
        "bleed_UNK",
        "elevation_False",
        "elevation_True",
        "elevation_UNK",
        "smoke_UNK",
        "drink_UNK",
        "pesticide_UNK",
        "gender_UNK",
        "skin_cancer_history_UNK",
        "cancer_history_UNK",
        "has_piped_water_UNK",
        "has_sewage_system_UNK",
        "fitspatrick_UNK",
        "region_UNK",
    ]

    METADATA_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def __init__(
        self,
        df: pd.DataFrame,
        transforms=ToTensor(),
        meta_ordered_features=None,
        image_folder=config.PAD_20_IMAGES_FOLDER,
    ):
        super().__init__()
        self.transforms = transforms
        self.targets = torch.tensor(df[PAD20.TARGET_NUMBER_COLUMN].values)
        self.images = df[PAD20.IMAGE_COLUMN].map(lambda x: image_folder / x)
        self.target_number_to_label = {}
        self.meta = torch.tensor(
            df[
                (
                    meta_ordered_features
                    if meta_ordered_features is not None
                    else (PAD20.CATEGORICAL_FEATURES + PAD20.NUMERICAL_FEATURES)
                )
            ].values
        )

        for i, row in df.iterrows():
            self.target_number_to_label[row[PAD20.TARGET_NUMBER_COLUMN]] = row[
                PAD20.TARGET_COLUMN
            ]

    def get_target_number_to_label(self):
        return self.target_number_to_label

    def __len__(self):
        return len(self.targets)

    def read_image(self, path, transforms):
        image = Image.open(path).convert("RGB")
        return transforms(np.asarray(image))

    def __getitem__(self, index):
        img_path = self.images.iloc[index]
        image = self.read_image(img_path, self.transforms)
        return image, self.meta[index], self.targets[index], Path(img_path).stem


class PAD20SentenceEmbedding(PAD20):

    def __init__(self, df, sentence_model, transforms=ToTensor()):
        super().__init__(df, transforms, [])
        self.sentence_model = sentence_model
        self.meta = self.get_metadata(df)

    def get_metadata(self, df: pd.DataFrame):
        return self.sentence_model.encode(
            df["sentence"].values, show_progress_bar=False
        )
