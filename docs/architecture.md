# System Architecture

This document explains the complete architecture of the Customer Churn Prediction & Behavioral Segmentation System — how every component connects, why each design decision was made, and how data flows through the system.

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                     │
│                                                                          │
│   main.py              src/api/app.py         docker-compose.yml        │
│   (Full Pipeline)      (REST API)             (Containerized)           │
└──────────┬─────────────────┬──────────────────────┬─────────────────────┘
           │                 │                      │
           ▼                 │                      │
┌──────────────────────┐     │                      │
│  LAYER 1: INGESTION  │     │                      │
│                      │     │                      │
│  generate_synthetic  │     │                      │
│  _data()             │     │                      │
│       │              │     │                      │
│       ▼              │     │                      │
│  save_to_csv()       │     │                      │
│       │              │     │                      │
│       ▼              │     │                      │
│  simulate_s3_upload()│     │                      │
└──────────┬───────────┘     │                      │
           │                 │                      │
           ▼                 │                      │
┌──────────────────────┐     │                      │
│  LAYER 2: TRANSFORM  │     │                      │
│                      │     │                      │
│  engineer_features() │     │                      │
│  ├─ tenure_buckets   │     │                      │
│  ├─ avg_monthly_spend│     │                      │
│  ├─ engagement_score │     │                      │
│  ├─ charge_ratio     │     │                      │
│  └─ support_intensity│     │                      │
│       │              │     │                      │
│       ▼              │     │                      │
│  preprocess()        │     │                      │
│  ├─ handle_missing   │     │                      │
│  ├─ encode_categorical│    │                      │
│  ├─ scale_features   │     │                      │
│  └─ validate         │     │                      │
│       │              │     │                      │
│       ├──► SQLite DB │     │                      │
│       └──► CSV Export│     │                      │
└──────────┬───────────┘     │                      │
           │                 │                      │
           ▼                 │                      │
┌──────────────────────┐     │                      │
│  LAYER 3: TRAINING   │     │                      │
│                      │     │                      │
│  split_data()        │     │                      │
│  (80/20 stratified)  │     │                      │
│       │              │     │                      │
│       ▼              │     │                      │
│  train_all_models()  │     │                      │
│  ├─ LogisticRegress. │     │                      │
│  ├─ RandomForest     │     │                      │
│  └─ XGBoost          │     │                      │
│       │              │     │                      │
│  Each uses:          │     │                      │
│  GridSearchCV(5-fold)│     │                      │
│       │              │     │                      │
│       ▼              │     │                      │
│  save_model()        │     │                      │
│  ├─ saved/ (latest)  │     │                      │
│  └─ versioned/       │     │                      │
└──────────┬───────────┘     │                      │
           │                 │                      │
           ▼                 │                      │
┌──────────────────────┐     │                      │
│  LAYER 4: EVALUATION │     │                      │
│                      │     │                      │
│  evaluate_all_models │     │                      │
│  ├─ Accuracy         │     │                      │
│  ├─ Precision/Recall │     │                      │
│  ├─ F1 Score         │     │                      │
│  ├─ ROC-AUC          │     │                      │
│  └─ Confusion Matrix │     │                      │
│       │              │     │                      │
│  select_best_model() │     │                      │
│  (by ROC-AUC)        │     │                      │
│       │              │     │                      │
│  Generates:          │     │                      │
│  ├─ ROC curves PNG   │     │                      │
│  ├─ Confusion PNGs   │     │                      │
│  ├─ Feature import.  │     │                      │
│  └─ comparison.csv   │     │                      │
└──────────┬───────────┘     │                      │
           │                 │                      │
           ▼                 │                      │
┌──────────────────────┐     │                      │
│  LAYER 5: CLUSTERING │     │                      │
│                      │     │                      │
│  run_clustering()    │     │                      │
│  ├─ elbow_method()   │     │                      │
│  ├─ silhouette()     │     │                      │
│  ├─ fit_kmeans(K=4)  │     │                      │
│  └─ interpret_       │     │                      │
│     segments()       │     │                      │
└──────────┬───────────┘     │                      │
           │                 │                      │
           ▼                 ▼                      │
┌──────────────────────────────────────┐            │
│  LAYER 6: OUTPUT                     │            │
│                                      │            │
│  BI Outputs:                         │            │
│  ├─ Tableau CSVs                     │            │
│  ├─ Executive KPI (JSON + CSV)       │            │
│  ├─ Segment profiles                 │            │
│  └─ Visualization PNGs              │            │
│                                      │            │
│  API (FastAPI):                      │◄───────────┘
│  ├─ POST /predict-churn             │
│  ├─ GET  /health                     │
│  └─ GET  /model-info                 │
│                                      │
│  Loads: xgboost.joblib + scaler     │
└──────────────────────────────────────┘
```

---

## Component Interactions

### How Components Talk to Each Other

The system follows a **linear pipeline pattern** for training and a **request-response pattern** for inference:

```
Training Flow (sequential):
ingestion.py → feature_engineering.py → preprocessing.py → train.py → evaluate.py → clustering.py → bi_output.py

Inference Flow (on-demand):
HTTP Request → app.py → loads xgboost.joblib + scaler.joblib → predict → HTTP Response
```

### Shared State Between Components

| Shared Resource | Producer | Consumers |
|----------------|----------|-----------|
| Raw CSV (`data/raw/`) | `ingestion.py` | `main.py` (reload for BI) |
| Processed CSV (`data/processed/`) | `preprocessing.py` | External tools, Tableau |
| SQLite DB (`data/processed/churn_data.db`) | `preprocessing.py` | SQL queries, external tools |
| StandardScaler (`models/saved/scaler.joblib`) | `preprocessing.py` via pipeline | `app.py`, `predict.py` |
| Trained models (`models/saved/*.joblib`) | `train.py` | `evaluate.py`, `app.py`, `predict.py` |
| Model metadata (`models/saved/*_metadata.json`) | `train.py` | `app.py` (`/model-info`) |
| Comparison report (`outputs/reports/`) | `evaluate.py` | `bi_output.py`, `main.py` |
| Segment profiles (`outputs/reports/`) | `clustering.py` | `bi_output.py` |

---

## Design Decisions

### Why Three Models?

| Model | Role | Strength |
|-------|------|----------|
| **Logistic Regression** | Baseline | Interpretable, fast, provides coefficient-based feature importance |
| **Random Forest** | Ensemble | Handles non-linear relationships, resistant to overfitting |
| **XGBoost** | Primary | Best predictive performance, handles imbalanced features well |

Training all three allows comparison and ensures the best model is selected objectively by ROC-AUC score.

### Why Synthetic Data?

The system generates its own data instead of requiring a specific dataset because:
1. **Reproducibility** — anyone can run the pipeline without external data dependencies
2. **Controlled quality** — we inject known patterns (contract type → churn, tenure → churn) to validate that models learn correctly
3. **Scale testing** — 100K records tests real-world performance characteristics
4. **Privacy** — no real customer data is needed

### Why SQLite (not PostgreSQL)?

SQLite was chosen for the default setup because:
- Zero configuration required
- No database server to install
- File-based, works everywhere
- The `.env` file includes Redshift connection variables for production upgrades

### Why Separate Scaler Saving?

The `StandardScaler` is saved independently (`scaler.joblib`) because:
- The API needs it at inference time to transform incoming features
- It must match the exact scaling used during training
- Model artifacts and preprocessing artifacts have different lifecycles

### Why Model Versioning?

Models are saved in two locations:
- `models/saved/` — always contains the **latest** version (overwritten each run)
- `models/versioned/` — timestamped copies for **audit trail** and **rollback**

This follows ML engineering best practices where you can always trace which model version produced which predictions.

---

## Data Flow Explanation

### Step 1: Raw Data Generation
100,000 synthetic customer records are created with 20 features. The churn label is generated using a **scoring algorithm** — each feature contributes a weighted score, which is converted to a probability via a sigmoid function. This creates realistic, learnable patterns.

### Step 2: Feature Engineering
Five new derived features are created from the raw data before any encoding or scaling. These features capture **business logic** that raw data alone doesn't express (e.g., "how many support tickets per month?" rather than just "total tickets").

### Step 3: Preprocessing
The data is cleaned (missing values imputed with medians), categorical features are one-hot encoded, and numeric features are standardized to zero mean and unit variance. The processed data is stored in both SQLite and CSV.

### Step 4: Model Training
The processed data is split 80/20 with stratified sampling to preserve the churn ratio. Each model is trained using 5-fold cross-validation with GridSearchCV to find optimal hyperparameters. All models and metadata are persisted.

### Step 5: Evaluation
Models are evaluated on the held-out test set. Metrics include accuracy, precision, recall, F1, and ROC-AUC. The best model is selected by ROC-AUC score. Confusion matrices and ROC curves are generated as PNG visualizations.

### Step 6: Clustering
K-Means clustering runs on the **original feature space** (not one-hot encoded) using 6 behavioral features. The elbow method and silhouette analysis determine the optimal number of clusters. Each cluster is labeled with a business-meaningful name.

### Step 7: BI Output
Tableau-ready CSVs, executive KPI summaries (JSON + CSV), and visualization charts are exported. These files are designed to be directly importable into BI tools without any additional transformation.

---

## Model Lifecycle

```
Development               Production                Monitoring
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ train.py    │    │ models/saved/   │    │ /health endpoint │
│             │    │                 │    │                  │
│ GridSearchCV│───►│ xgboost.joblib  │───►│ Status check     │
│ 5-fold CV   │    │ scaler.joblib   │    │ Model version    │
│             │    │ *_metadata.json │    │                  │
│ Versioned   │    │                 │    │ /model-info      │
│ copy saved  │    │ Loaded by       │    │ Training params  │
│             │    │ FastAPI startup │    │ Best CV score    │
└─────────────┘    └─────────────────┘    └──────────────────┘
       │
       ▼
┌─────────────┐
│ models/     │
│ versioned/  │
│             │
│ Timestamped │
│ archive for │
│ rollback    │
└─────────────┘
```

---

## Technology Stack Justification

| Technology | Why It's Used |
|-----------|---------------|
| **Python 3.11** | Industry standard for ML, rich ecosystem |
| **Pandas** | DataFrame operations, data manipulation |
| **NumPy** | Numerical computations, array operations |
| **scikit-learn** | Preprocessing, Logistic Regression, Random Forest, K-Means, metrics |
| **XGBoost** | State-of-the-art gradient boosting classifier |
| **Matplotlib** | Visualization (confusion matrices, ROC curves, feature importance) |
| **FastAPI** | High-performance async API framework with auto-generated docs |
| **Pydantic** | Request/response validation and serialization |
| **SQLite** | Lightweight SQL database for processed data storage |
| **joblib** | Efficient model serialization (better than pickle for NumPy arrays) |
| **python-dotenv** | Environment variable management |
| **Docker** | Containerized deployment |
| **Uvicorn** | ASGI server for FastAPI |
