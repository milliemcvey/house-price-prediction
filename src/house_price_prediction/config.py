from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_house_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "linear_regression_model.pkl"

RANDOM_SEED = 42
