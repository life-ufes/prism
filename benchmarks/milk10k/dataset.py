import pandas as pd
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MILK10K(Dataset):
    CLINICAL_IMAGE_COLUMN = "image_clinical"
    DERMATOSCOPIC_IMAGE_COLUMN = "image_dermoscopic"
    LESION_ID_COLUMN = "lesion_id"
    LABELS = [
        "AKIEC",
        "BCC",
        "BEN_OTH",
        "BKL",
        "DF",
        "INF",
        "MAL_OTH",
        "MEL",
        "NV",
        "SCCKA",
        "VASC",
    ]
    TARGET_COLUMN = "diagnostic"
    TARGET_NUMBER_COLUMN = "diagnostic_number"
    NUMERICAL_FEATURES = [
        "age_approx",
        "dermoscopic_ulceration_crust",
        "dermoscopic_hair",
        "dermoscopic_vasculature_vessels",
        "dermoscopic_erythema",
        "dermoscopic_pigmented",
        "dermoscopic_gel_water_drop_fluid_dermoscopy_liquid",
        "dermoscopic_skin_markings_pen_ink_purple_pen",
    ]
    IMAGE_TYPE_COLUMN = "image_type"
    RAW_CATEGORICAL_FEATURES = ["sex", "skin_tone_class", "site"]
    CATEGORICAL_FEATURES = [
        "sex_female",
        "sex_male",
        "skin_tone_class_1",
        "skin_tone_class_2",
        "skin_tone_class_3",
        "skin_tone_class_4",
        "skin_tone_class_5",
        "site_foot",
        "site_genital",
        "site_hand",
        "site_head_neck_face",
        "site_lower_extremity",
        "site_trunk",
        "site_upper_extremity",
    ]

    METADATA_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def __init__(
        self,
        df,
        clinical_transforms=ToTensor(),
        dermoscopic_transforms=ToTensor(),
        meta_ordered_features=None,
    ):
        super().__init__()
        self.clinical_transforms = clinical_transforms
        self.dermoscopic_transforms = dermoscopic_transforms
        self.has_targets = MILK10K.TARGET_NUMBER_COLUMN in df
        self.clinical_images = df[MILK10K.CLINICAL_IMAGE_COLUMN]
        self.dermoscopic_images = df[MILK10K.DERMATOSCOPIC_IMAGE_COLUMN]
        self.target_number_to_label = {}
        features = (
            meta_ordered_features
            if meta_ordered_features is not None
            else (MILK10K.CATEGORICAL_FEATURES + MILK10K.NUMERICAL_FEATURES)
        )
        self.meta = torch.tensor(df[features].values, dtype=torch.float32)
        self.lesion_ids = df[MILK10K.LESION_ID_COLUMN].values

        if self.has_targets:
            self.targets = (
                torch.tensor(df[MILK10K.TARGET_NUMBER_COLUMN].values, dtype=torch.long)
                if self.has_targets
                else None
            )
            for _, row in df.iterrows():
                self.target_number_to_label[row[MILK10K.TARGET_NUMBER_COLUMN]] = row[
                    MILK10K.TARGET_COLUMN
                ]

    def get_target_number_to_label(self):
        return {
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
        return self.meta.shape[0]

    def read_image(self, path, transforms):
        image = Image.open(path).convert("RGB")
        return transforms(np.asarray(image))

    def __getitem__(self, index):
        return (
            self.read_image(
                self.dermoscopic_images.iloc[index], self.dermoscopic_transforms
            ),
            self.meta[index],
            self.targets[index] if self.has_targets else -1,
            self.lesion_ids[index],
        )


class MILK10KSentenceEmbedding(MILK10K):

    def __init__(
        self,
        df,
        sentence_model,
        clinical_transforms=ToTensor(),
        dermoscopic_transforms=ToTensor(),
    ):
        super().__init__(df, clinical_transforms, dermoscopic_transforms, [])
        self.sentence_model = sentence_model
        self.meta = self.get_metadata(df)

    def get_metadata(self, df: pd.DataFrame):
        return self.sentence_model.encode(
            df["sentence"].values, show_progress_bar=False
        )
