import config
import numpy as np
import pandas as pd
from benchmarks.milk10k.dataset import MILK10K
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def generate_sentence(df: pd.DataFrame) -> pd.DataFrame:

	df = df.reset_index(drop=True)
	for col in df.select_dtypes(include="category").columns:
		if "unknown" not in df[col].cat.categories:
			df[col] = df[col].cat.add_categories("unknown")

	site_name_map = {
		"head_neck_face": "Head, Neck, or Face",
		"lower_extremity": "Lower Extremity",
		"upper_extremity": "Upper Extremity",
		"trunk": "Torso",
		"foot": "Foot",
		"genital": "Genital Area",
		"hand": "Hand",
	}

	skin_tone_name_map = {
		1: "Dark",
		2: "Medium-Dark",
		3: "Medium",
		4: "Medium-Light",
		5: "Light",
	}

	df["skin_tone_class"] = (
		df["skin_tone_class"].map(skin_tone_name_map).fillna("unknown")
	)

	df["site"] = df["site"].map(site_name_map).fillna("unknown")

	df["age_approx"] = df["age_approx"].fillna("unknown")

	field_to_label = {
		"age_approx": "Approximate Age",
		"dermoscopic_ulceration_crust": "Ulceration or Crust Probability",
		"dermoscopic_hair": "Hair Probability",
		"dermoscopic_vasculature_vessels": "Vasculature or Vessels Probability",
		"dermoscopic_erythema": "Erythema Probability",
		"dermoscopic_pigmented": "Pigmented Probability",
		"dermoscopic_gel_water_drop_fluid_dermoscopy_liquid": "Gel, Water, or Fluid Probability",
		"dermoscopic_skin_markings_pen_ink_purple_pen": "Skin Markings or Pen Ink Probability",
		"sex": "Sex",
		"skin_tone_class": "Skin Tone",
		"site": "Anatomical Site",
	}

	sentences = []

	prob_features = [
		f for f in df.columns if f.startswith("clinical") or f.startswith("dermoscopic")
	]

	features = [
		"age_approx",
		"dermoscopic_ulceration_crust",
		"dermoscopic_hair",
		"dermoscopic_vasculature_vessels",
		"dermoscopic_erythema",
		"dermoscopic_pigmented",
		"dermoscopic_gel_water_drop_fluid_dermoscopy_liquid",
		"dermoscopic_skin_markings_pen_ink_purple_pen",
		"sex",
		"skin_tone_class",
		"site",
	]
	for _, row in df.iterrows():
		anamnese = "Patient History: "
		for col in features:
			if row[col] != "unknown":
				value = (
					str(row[col]).replace(".0", "")
					if col not in prob_features
					else str(int(row[col] * 100)) + "%"
				)
				anamnese += f"{field_to_label[col]}: {value.lower()}, "
		anamnese = anamnese[:-2] + "."
		sentences.append(anamnese)
	df["sentence"] = pd.Series(sentences)
	return df[
		[
			MILK10K.LESION_ID_COLUMN,
			MILK10K.TARGET_COLUMN,
			MILK10K.TARGET_NUMBER_COLUMN,
			MILK10K.DERMATOSCOPIC_IMAGE_COLUMN,
			MILK10K.CLINICAL_IMAGE_COLUMN,
			"folder",
			"sentence",
		]
	]


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
	)  # MILK10k dataset has no patient-level information.

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

	df = generate_sentence(df)
	output_path.parent.mkdir(exist_ok=True)

	df.to_csv(output_path)

	print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
	for features_path, labels_path, images_folder, output_path in [
		(
			config.MILK10K_TRAIN_RAW_METADATA,
			config.MILK10K_TRAIN_LABELS,
			config.MILK10K_TRAIN_IMAGES_FOLDER,
			config.MILK10K_TRAIN_SENTENCE,
		),
	]:
		_preprocess(features_path, labels_path, images_folder, output_path)
