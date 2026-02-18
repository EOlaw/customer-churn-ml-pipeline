# Configuration Guide

This document explains every configurable setting in the system — environment variables, application constants, logging, database configuration, and how to customize the pipeline for different environments.

---

## Configuration Architecture

All configuration lives in one place: `src/config/settings.py`. Every module imports from this file — there are no hardcoded paths or magic numbers scattered across the codebase.

```
.env (secrets & overrides)
    │
    ▼ python-dotenv
src/config/settings.py (central config)
    │
    ├──► src/data/ingestion.py
    ├──► src/data/preprocessing.py
    ├──► src/models/train.py
    ├──► src/models/evaluate.py
    ├──► src/models/clustering.py
    ├──► src/api/app.py
    └──► src/utils/logger.py
```

---

## Environment Variables (.env)

The `.env` file contains secrets and environment-specific settings. It is **not committed to git** (only `.env.example` is committed).

### Setup

```bash
cp .env.example .env
# Edit .env with your values
```

### All Environment Variables

#### AWS Credentials (Simulated)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | `""` | AWS access key for S3/Redshift |
| `AWS_SECRET_ACCESS_KEY` | `""` | AWS secret key |
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET_NAME` | `churn-ml-pipeline` | S3 bucket name for data storage |

**Note:** These are used for simulation only. In production, replace with real credentials or use IAM roles.

#### Redshift Configuration (Simulated)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDSHIFT_HOST` | `localhost` | Redshift cluster endpoint |
| `REDSHIFT_PORT` | `5439` | Redshift port (standard) |
| `REDSHIFT_DB` | `churn_analytics` | Database name |

**Note:** The current system uses SQLite. To switch to Redshift, update the database connection in `preprocessing.py`.

#### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |

#### Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VERSION` | `1.0.0` | Semantic version for trained models |

Increment this when retraining with new data or parameters:
- **Patch** (1.0.1): Bug fix in preprocessing
- **Minor** (1.1.0): New features added
- **Major** (2.0.0): New model architecture or training data

#### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

---

## Application Constants

These are set in `src/config/settings.py` and are **not overridable via environment variables** (change them in the source code):

### Data Generation

| Constant | Value | Description |
|----------|-------|-------------|
| `N_SAMPLES` | `100_000` | Number of synthetic records to generate |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |
| `TEST_SIZE` | `0.2` | Fraction of data used for testing (20%) |

### Model Training

| Constant | Value | Description |
|----------|-------|-------------|
| `CV_FOLDS` | `5` | Number of cross-validation folds |
| `SCORING_METRIC` | `"roc_auc"` | Metric used for model selection |

### To Change the Number of Samples

Edit `src/config/settings.py`:
```python
N_SAMPLES = 200_000  # Generate 200K records instead of 100K
```

Or override in code:
```python
from src.data.ingestion import generate_synthetic_data
df = generate_synthetic_data(n_samples=50_000)  # Smaller dataset for testing
```

---

## Directory Paths

All paths are derived from `PROJECT_ROOT` using `pathlib.Path`:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # customer-churn-ml-pipeline/

# Data
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR  = DATA_DIR / "external"
S3_SIMULATION_DIR  = DATA_DIR / "s3_simulation"

# Models
MODELS_DIR         = PROJECT_ROOT / "models"
SAVED_MODELS_DIR   = MODELS_DIR / "saved"
VERSIONED_MODELS_DIR = MODELS_DIR / "versioned"

# Outputs
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
TABLEAU_DIR       = OUTPUTS_DIR / "tableau"
REPORTS_DIR       = OUTPUTS_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"
```

All directories are created automatically when first accessed (each module calls `mkdir(parents=True, exist_ok=True)`).

---

## Database Configuration

### SQLite (Default)

```python
DATABASE_PATH = PROCESSED_DATA_DIR / "churn_data.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
```

The database is created automatically at `data/processed/churn_data.db` when the preprocessing step runs. It contains one table: `customer_data`.

### Switching to PostgreSQL

To use PostgreSQL instead of SQLite:

1. Install the driver: `pip install psycopg2-binary`
2. Update `.env`:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/churn_db
   ```
3. Update `settings.py`:
   ```python
   DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
   ```
4. Update `preprocessing.py` to use SQLAlchemy:
   ```python
   from sqlalchemy import create_engine
   engine = create_engine(DATABASE_URL)
   df.to_sql("customer_data", engine, if_exists="replace", index=False)
   ```

### Switching to Amazon Redshift

1. Install: `pip install sqlalchemy-redshift redshift_connector`
2. Use the Redshift environment variables already defined in `.env`
3. Connection string: `redshift+redshift_connector://user:pass@host:5439/db`

---

## Logging Configuration

**File:** `src/utils/logger.py`

### How Logging Works

Every module creates a logger using:
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```

This creates a logger named after the module (e.g., `src.data.ingestion`) with:
- **File handler:** Writes to `logs/pipeline.log`
- **Stream handler:** Prints to stdout (console)

### Log Format

```
2026-02-17 22:07:42 | src.data.ingestion | INFO | Generating 100000 synthetic customer records...
│                      │                    │      └── Message
│                      │                    └── Log level
│                      └── Module name
└── Timestamp
```

### Log Levels

| Level | When to Use | Example |
|-------|------------|---------|
| `DEBUG` | Detailed diagnostic information | Variable values, loop iterations |
| `INFO` | Confirmation that things work as expected | "Loaded 100000 records" |
| `WARNING` | Something unexpected but not an error | "Models not found" |
| `ERROR` | Something failed | "Validation error: Missing columns" |
| `CRITICAL` | The system is unusable | Database connection failure |

### Changing Log Level

In `.env`:
```
LOG_LEVEL=DEBUG    # See everything
LOG_LEVEL=WARNING  # Only problems
```

---

## Model Hyperparameter Configuration

Model hyperparameter search spaces are defined in `src/models/train.py` as the `MODEL_CONFIGS` dictionary:

```python
MODEL_CONFIGS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "C": [0.01, 0.1, 1.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
}
```

### To Add a New Model

Add an entry to `MODEL_CONFIGS`:
```python
"gradient_boosting": {
    "estimator": GradientBoostingClassifier(random_state=42),
    "param_grid": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
},
```

The training pipeline will automatically pick it up and include it in evaluation.

### To Expand the Search Space

Add more values to explore:
```python
"learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],  # More granular
```

**Warning:** Each additional value multiplies the number of fits. A grid of 24 combinations × 5 folds = 120 fits. Doubling the grid size doubles training time.

---

## Clustering Configuration

Clustering features and parameters are defined in `src/models/clustering.py`:

```python
CLUSTERING_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets", "monthly_minutes_used", "data_usage_gb",
]
```

To change the number of clusters, modify the call in `training_pipeline.py`:
```python
labels, segment_profile = run_clustering(raw_df, n_clusters=5)  # Change from 4 to 5
```

---

## Validation Configuration

Required columns for raw data validation are defined in `src/utils/validation.py`:

```python
REQUIRED_COLUMNS = [
    "customer_id", "gender", "age", "tenure_months", "contract_type",
    "monthly_charges", "total_charges", "payment_method",
    "num_support_tickets", "internet_service", "online_security",
    "tech_support", "streaming_tv", "streaming_movies",
    "paperless_billing", "num_dependents", "partner",
    "monthly_minutes_used", "data_usage_gb", "churn",
]
```

If using a real dataset with different column names, update this list.

---

## Risk Level Thresholds

Risk levels for churn predictions are defined in `src/models/predict.py` and `src/api/app.py`:

```python
if proba >= 0.7:
    risk_level = "High"
elif proba >= 0.4:
    risk_level = "Medium"
else:
    risk_level = "Low"
```

Adjust these thresholds based on business requirements:
- **Conservative** (catch more churners): High >= 0.5, Medium >= 0.3
- **Aggressive** (fewer false alarms): High >= 0.8, Medium >= 0.5
