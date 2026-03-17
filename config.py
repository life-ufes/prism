from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = Path('.')
BENCHMARKS = ROOT / 'benchmarks'

##################################################################################################################
# PAD-20
PAD_20_BENCHMARK = BENCHMARKS / 'pad20'
PAD_20_PATH = Path("/home/pedrobouzon/life/datasets/pad-ufes-20")
PAD_20_IMAGES_FOLDER = PAD_20_PATH / "images"
PAD_20_ONE_HOT_ENCODED = PAD_20_BENCHMARK / 'data' / "pad-ufes-20-one-hot.csv"
PAD_20_RAW_METADATA = PAD_20_PATH / "metadata.csv"
PAD_20_SENTENCE = PAD_20_BENCHMARK / 'data' / "pad-ufes-20-sentence.csv"
PAD_20_BAYESIAN_DATA = PAD_20_BENCHMARK / 'data' / 'bayesian'

##################################################################################################################
# MILK10K
MILK10K_PATH = Path("/home/pedrobouzon/life/datasets/MILK10K")
MILK10K_BENCHMARK = BENCHMARKS / 'milk10k'
MILK10K_TRAIN_IMAGES_FOLDER = MILK10K_PATH / 'MILK10k_Training_Input'
MILK10K_TRAIN_ONE_HOT_ENCODED = MILK10K_BENCHMARK / 'data' / 'train-one-hot-encoded.csv'
MILK10K_TRAIN_SENTENCE = MILK10K_BENCHMARK / 'data' / 'train-sentence.csv'
MILK10K_TRAIN_ONE_HOT_ENCODED_WITH_MIDAS = MILK10K_BENCHMARK / 'data' / 'train-one-hot-encoded-midas.csv'
MILK10K_TRAIN_RAW_METADATA = MILK10K_PATH / "MILK10k_Training_Metadata.csv"
MILK10K_TRAIN_LABELS = MILK10K_PATH / "MILK10k_Training_GroundTruth.csv"
MILK10K_BAYESIAN_DATA = MILK10K_BENCHMARK / 'data' / 'bayesian'