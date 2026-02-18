"""
Project configuration and settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
S3_SIMULATION_DIR = DATA_DIR / "s3_simulation"

MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
VERSIONED_MODELS_DIR = MODELS_DIR / "versioned"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLEAU_DIR = OUTPUTS_DIR / "tableau"
REPORTS_DIR = OUTPUTS_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

LOGS_DIR = PROJECT_ROOT / "logs"

# Database
DATABASE_PATH = PROCESSED_DATA_DIR / "churn_data.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# AWS simulation credentials (from .env)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "churn-ml-pipeline")
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST", "localhost")
REDSHIFT_PORT = os.getenv("REDSHIFT_PORT", "5439")
REDSHIFT_DB = os.getenv("REDSHIFT_DB", "churn_analytics")

# Data generation
N_SAMPLES = 100_000
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Model parameters
CV_FOLDS = 5
SCORING_METRIC = "roc_auc"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "pipeline.log"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model version
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
