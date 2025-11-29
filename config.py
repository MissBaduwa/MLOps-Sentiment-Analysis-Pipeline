# config.py
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(exist_ok=True)

# MLflow settings
MLFLOW_TRACKING_URI = "file:" + str(PROJECT_ROOT / "mlruns")
MLFLOW_EXPERIMENT_NAME = "sentiment_analysis"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2