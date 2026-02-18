# Project Structure Guide

This document explains every folder and file in the project — what it contains, why it exists, and what it produces.

---

## Complete Directory Tree

```
customer-churn-ml-pipeline/
│
├── data/                          ← All data files (raw, processed, simulated cloud)
│   ├── raw/                       ← Original unprocessed data
│   │   └── customer_churn_raw.csv
│   ├── processed/                 ← Cleaned, encoded, scaled data
│   │   ├── churn_data.db          ← SQLite database
│   │   └── customer_churn_processed.csv
│   ├── external/                  ← External datasets (empty by default)
│   └── s3_simulation/             ← Simulated AWS S3 bucket
│       ├── raw/
│       │   └── customer_churn_raw.csv
│       ├── processed/
│       └── models/
│
├── models/                        ← Trained model artifacts
│   ├── saved/                     ← Latest versions (used by API)
│   │   ├── logistic_regression.joblib
│   │   ├── logistic_regression_metadata.json
│   │   ├── random_forest.joblib
│   │   ├── random_forest_metadata.json
│   │   ├── xgboost.joblib
│   │   ├── xgboost_metadata.json
│   │   ├── scaler.joblib
│   │   ├── kmeans_model.joblib
│   │   └── clustering_scaler.joblib
│   └── versioned/                 ← Timestamped copies for audit trail
│       ├── xgboost_v1.0.0_20260217_220843.joblib
│       ├── random_forest_v1.0.0_20260217_220831.joblib
│       └── logistic_regression_v1.0.0_20260217_220745.joblib
│
├── outputs/                       ← All pipeline outputs
│   ├── tableau/                   ← Tableau-ready CSV exports
│   │   ├── churn_analysis_tableau.csv
│   │   ├── segment_summary.csv
│   │   └── executive_kpi.csv
│   ├── reports/                   ← Evaluation and KPI reports
│   │   ├── model_comparison.csv
│   │   ├── model_comparison.json
│   │   ├── segment_profiles.csv
│   │   └── executive_kpi_summary.json
│   └── visualizations/            ← Charts and plots (PNG)
│       ├── confusion_matrix_logistic_regression.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── roc_curves_comparison.png
│       ├── feature_importance_xgboost.png
│       ├── elbow_method.png
│       ├── silhouette_analysis.png
│       └── churn_distribution.png
│
├── src/                           ← Source code (all Python modules)
│   ├── __init__.py
│   ├── config/                    ← Configuration and settings
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data/                      ← Data pipeline modules
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/                    ← ML model modules
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── clustering.py
│   ├── pipelines/                 ← Pipeline orchestration
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   ├── api/                       ← FastAPI application
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/                     ← Shared utilities
│       ├── __init__.py
│       ├── logger.py
│       ├── validation.py
│       └── bi_output.py
│
├── tests/                         ← Test suite
│   ├── __init__.py
│   └── test_pipeline.py
│
├── docs/                          ← Documentation
│   ├── INDEX.md
│   ├── architecture.md
│   ├── data-pipeline.md
│   ├── ml-models.md
│   ├── clustering-segmentation.md
│   ├── api-reference.md
│   ├── deployment.md
│   ├── configuration.md
│   ├── bi-outputs.md
│   └── project-structure.md
│
├── logs/                          ← Pipeline execution logs
│   └── pipeline.log
│
├── notebooks/                     ← Jupyter notebooks (for exploration)
│
├── main.py                        ← Main entry point
├── requirements.txt               ← Python dependencies
├── Dockerfile                     ← Docker image definition
├── docker-compose.yml             ← Multi-container orchestration
├── .env.example                   ← Environment variable template
├── .gitignore                     ← Git ignore rules
└── README.md                      ← Project overview
```

---

## Folder-by-Folder Explanation

### `data/` — Data Storage

This folder stores all data at every stage of the pipeline. It is **gitignored** (except the directory structure) to prevent large files from being committed.

#### `data/raw/`
**Contains:** The original generated dataset before any processing.
**Key file:** `customer_churn_raw.csv` — 100,000 rows, 20 columns, includes missing values.
**Created by:** `src/data/ingestion.py`
**Used by:** `main.py` (reloaded for BI outputs), feature engineering

#### `data/processed/`
**Contains:** Cleaned, encoded, and scaled data ready for model training.
**Key files:**
- `customer_churn_processed.csv` — 100,000 rows, 36 columns (after encoding), no missing values, scaled
- `churn_data.db` — SQLite database containing the same data in table `customer_data`

**Created by:** `src/data/preprocessing.py`
**Used by:** Model training, external SQL queries

**How to query the database:**
```bash
sqlite3 data/processed/churn_data.db "SELECT COUNT(*) FROM customer_data WHERE churn = 1;"
```

#### `data/external/`
**Contains:** External datasets that could supplement the pipeline (empty by default).
**Purpose:** Placeholder for real-world data sources like demographic data, economic indicators, or third-party churn benchmarks.

#### `data/s3_simulation/`
**Contains:** A local directory structure that mimics an AWS S3 bucket.
**Purpose:** Demonstrates cloud-ready data organization without requiring actual AWS credentials.
**Structure mirrors:** `s3://churn-ml-pipeline/raw/`, `s3://churn-ml-pipeline/processed/`, `s3://churn-ml-pipeline/models/`

---

### `models/` — Model Artifacts

#### `models/saved/`
**Contains:** The **latest** trained models and preprocessing artifacts. These are the files loaded by the API for inference.

| File | Size | Description |
|------|------|-------------|
| `xgboost.joblib` | ~234 KB | Best classifier (XGBoost) |
| `random_forest.joblib` | ~227 MB | Random Forest (large due to many trees) |
| `logistic_regression.joblib` | ~2 KB | Logistic Regression (small, just coefficients) |
| `scaler.joblib` | ~2 KB | StandardScaler for classification features |
| `kmeans_model.joblib` | ~357 KB | K-Means clustering model |
| `clustering_scaler.joblib` | ~1 KB | StandardScaler for clustering features |
| `*_metadata.json` | <1 KB each | Training parameters, CV scores, timestamps |

**Important:** The API loads `xgboost.joblib` and `scaler.joblib` at startup. Both must exist for predictions to work.

#### `models/versioned/`
**Contains:** Timestamped copies of every trained model. Used for:
- **Audit trail:** Know exactly which model produced which predictions
- **Rollback:** Restore a previous model version if a new one underperforms
- **Comparison:** Compare model versions over time

**Naming convention:** `{model_name}_v{version}_{timestamp}.joblib`

---

### `outputs/` — Pipeline Outputs

#### `outputs/tableau/`
**Contains:** CSV files designed to be imported directly into Tableau Desktop or Tableau Public.

| File | Rows | Description |
|------|------|-------------|
| `churn_analysis_tableau.csv` | 100,000 | Full dataset with segment labels |
| `segment_summary.csv` | 4 | Segment profiles with business labels |
| `executive_kpi.csv` | 1 | Top-level KPIs for dashboard |

#### `outputs/reports/`
**Contains:** Machine-readable evaluation reports.

| File | Format | Description |
|------|--------|-------------|
| `model_comparison.csv` | CSV | Accuracy, ROC-AUC, F1 for all 3 models |
| `model_comparison.json` | JSON | Full metrics including confusion matrices |
| `segment_profiles.csv` | CSV | Cluster profiles with business labels |
| `executive_kpi_summary.json` | JSON | Complete KPI package (business + model + segments) |

#### `outputs/visualizations/`
**Contains:** All chart images generated by the pipeline.

| File | Description |
|------|-------------|
| `confusion_matrix_*.png` | Confusion matrix for each model (3 files) |
| `roc_curves_comparison.png` | ROC curves for all models on one chart |
| `feature_importance_xgboost.png` | Top 15 features bar chart |
| `elbow_method.png` | K-Means elbow method chart |
| `silhouette_analysis.png` | Silhouette score analysis chart |
| `churn_distribution.png` | 3-panel: churn distribution, charges by churn, tenure by churn |

---

### `src/` — Source Code

The source code is organized by **responsibility**:

#### `src/config/`
**Central configuration.** `settings.py` defines all paths, constants, and environment variables. Every other module imports from here — no hardcoded paths anywhere.

#### `src/data/`
**Data pipeline modules.** Three files handle the full ETL:
1. `ingestion.py` — Generate/load data, simulate S3
2. `feature_engineering.py` — Create 5 derived features
3. `preprocessing.py` — Clean, encode, scale, store

#### `src/models/`
**ML modules.** Four files cover the full model lifecycle:
1. `train.py` — Train 3 classifiers with GridSearchCV
2. `evaluate.py` — Metrics, confusion matrices, ROC curves
3. `predict.py` — Single and batch prediction functions
4. `clustering.py` — K-Means segmentation with business labels

#### `src/pipelines/`
**Orchestration.** `training_pipeline.py` calls all modules in sequence to run the complete pipeline in one command.

#### `src/api/`
**REST API.** `app.py` provides the FastAPI inference endpoint with Pydantic validation.

#### `src/utils/`
**Shared utilities.** Three cross-cutting concerns:
1. `logger.py` — Centralized logging (file + stdout)
2. `validation.py` — Data quality checks
3. `bi_output.py` — Tableau/BI export generation

---

### `tests/` — Test Suite

Contains `test_pipeline.py` with 11 unit tests across 4 test classes covering data ingestion, feature engineering, preprocessing, and validation. Run with `pytest tests/ -v`.

---

### `logs/` — Execution Logs

Contains `pipeline.log` with timestamped, structured logs from every pipeline run. Useful for debugging and auditing.

---

### `notebooks/` — Jupyter Notebooks

Empty by default. Intended for:
- Exploratory data analysis (EDA)
- Model experimentation
- Visualization prototyping
- Ad-hoc analysis

---

### Root Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — runs full pipeline + BI output generation |
| `requirements.txt` | Python dependencies with minimum versions |
| `Dockerfile` | Container definition for the API |
| `docker-compose.yml` | Multi-service orchestration (API + training) |
| `.env.example` | Template for environment variables |
| `.gitignore` | Excludes data, models, outputs, logs, venv, .env from git |
| `README.md` | Project overview, quick start, and results |

---

## What's Gitignored

The `.gitignore` excludes:
- `data/raw/`, `data/processed/`, `data/s3_simulation/` — Large data files
- `models/saved/`, `models/versioned/` — Large model files
- `outputs/` — Generated outputs
- `logs/` — Log files
- `.env` — Secrets (only `.env.example` is committed)
- `venv/` — Virtual environment
- `__pycache__/` — Python bytecode
- `.DS_Store` — macOS metadata
