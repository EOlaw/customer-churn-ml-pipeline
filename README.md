# 🔄 Customer Churn ML Pipeline — Predictive Analytics & Behavioral Segmentation Platform

> A production-ready, end-to-end ML pipeline that predicts customer churn probability, segments customers into behavioral cohorts, quantifies revenue at risk, and exports executive-ready BI artifacts — all powered by a multi-model ensemble, automated hyperparameter tuning, and a clean layered Python architecture.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/language-Python%203.11-111827?style=for-the-badge" />
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI%20%2B%20Pydantic-0F766E?style=for-the-badge" />
  <img alt="ML" src="https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Logistic%20Regression-7C3AED?style=for-the-badge" />
  <img alt="BI" src="https://img.shields.io/badge/BI-Tableau--Ready%20Exports%20%26%20Executive%20KPIs-1D4ED8?style=for-the-badge" />
</p>

---

## 🔍 Problem

Customer churn is one of the most expensive problems in subscription and service businesses — but most companies only discover it after it has already happened. Without a reliable early-warning system, retention teams operate reactively, wasting budget on customers who would have stayed, while missing the ones who are genuinely at risk.

**Who is affected:**
- Revenue teams flying blind on which customers are about to leave and how much ARR is at risk
- Customer success teams who rely on gut instinct and lagging indicators (NPS, renewal dates) instead of predictive signals
- Analytics teams that have the raw data but no automated pipeline to turn it into actionable churn scores
- Executives who need segment-level visibility into their customer base to make retention investment decisions

**Why it matters:**
- Acquiring a new customer costs 5–7× more than retaining an existing one — every churn that goes undetected is a compounding loss
- Standard reporting (usage dashboards, ticket counts) shows what already happened, not what will happen next
- Manual scoring approaches don't scale past a few hundred customers and can't run continuously
- Without behavioral segmentation, retention campaigns are one-size-fits-all and low ROI

---

## 💡 Solution

Built a **production-ready churn intelligence pipeline** that ingests raw customer data, engineers business-meaningful features, trains and compares three ML models under cross-validation, selects the best performer by ROC-AUC, runs K-Means behavioral segmentation, and exports a full suite of BI artifacts and a live REST API for real-time inference.

- **Data ingestion layer** (`src/data/ingestion.py`) generates 100K+ synthetic customer records with realistic churn patterns, validates schema, and simulates S3 upload and storage
- **Feature engineering** (`src/data/feature_engineering.py`) derives five business-logic features — tenure buckets, average monthly spend, engagement score, charge ratio, and support intensity — that capture signals the raw data doesn't express
- **Preprocessing pipeline** (`src/data/preprocessing.py`) handles missing values via median imputation, one-hot encodes categoricals, standardizes numeric features, and persists data to both SQLite and CSV
- **Model training** (`src/models/train.py`) trains Logistic Regression, Random Forest, and XGBoost under 5-fold GridSearchCV and saves every artifact with versioned timestamped copies for rollback
- **Evaluation engine** (`src/models/evaluate.py`) computes accuracy, precision, recall, F1, and ROC-AUC on the held-out test set and generates confusion matrices, ROC curves, and feature importance charts
- **Clustering** (`src/models/clustering.py`) runs K-Means with elbow method and silhouette analysis to identify four behavioral customer segments with business labels
- **BI output layer** (`src/utils/bi_output.py`) exports Tableau-ready CSVs, executive KPI summaries in JSON and CSV, and visualization artifacts directly importable into any BI tool
- **FastAPI inference server** (`src/api/app.py`) loads the best model at startup and serves real-time single-customer and batch churn predictions over a typed REST interface

---

## 🧠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML / Modeling** | scikit-learn (LogisticRegression, RandomForest, K-Means, GridSearchCV), XGBoost |
| **Data Processing** | Pandas, NumPy |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Model Serialization** | joblib |
| **Database** | SQLite (local) — Redshift-ready via env config |
| **Storage** | Local filesystem with S3-simulated directory structure |
| **Visualization** | Matplotlib |
| **BI Exports** | Tableau-ready CSV artifacts, executive KPI JSON |
| **Deployment** | Docker, docker-compose |
| **Config** | python-dotenv |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  CSV / Simulated S3  ──►  Raw Data Storage  ──►  Validation    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    ETL PIPELINE                                  │
│  Missing Values ──► Feature Engineering ──► Encoding ──► Scaling│
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │   SQLite Storage    │                             │
│              │   S3 Simulation     │                             │
│              └─────────────────────┘                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                MODEL TRAINING & EVALUATION                      │
│                                                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │ Logistic Reg.  │  │ Random Forest│  │    XGBoost      │     │
│  │  (baseline)    │  │              │  │  (primary)      │     │
│  └────────────────┘  └──────────────┘  └─────────────────┘     │
│                                                                  │
│  GridSearchCV  ──►  5-Fold Cross-Validation  ──►  Best Model   │
│  ROC-AUC · Precision · Recall · Confusion Matrix                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              CLUSTERING & SEGMENTATION                          │
│  K-Means  ──►  Elbow Method  ──►  Silhouette Analysis          │
│  Segment Interpretation  ──►  Business Labels                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  FastAPI      │  │  Tableau CSVs  │  │  Executive KPIs  │   │
│  │ /predict-churn│  │  segment_summary│  │  JSON + CSV      │   │
│  │ /health       │  │  executive_kpi  │  │  Revenue at risk │   │
│  │ /model-info   │  └────────────────┘  └──────────────────┘   │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**
- Raw records → feature-engineered → preprocessed → stored in SQLite and CSV
- Three models trained in parallel under GridSearchCV → evaluated on held-out test set → best selected by ROC-AUC
- K-Means runs on original feature space → four behavioral segments with business labels
- All outputs (models, metrics, visualizations, BI exports) stored in versioned, traceable paths

---

## ⚙️ How It Works

1. **Ingestion** — `ingestion.py` generates 100K synthetic customer records with 20 features and a scoring-algorithm-derived churn label (weighted sigmoid of risk factors). Records are saved to CSV and mirrored into an S3-simulated directory structure.

2. **Feature Engineering** — `feature_engineering.py` derives five domain features before any encoding or scaling: `tenure_bucket` (lifecycle stage), `avg_monthly_spend` (total charges normalized by tenure), `engagement_score` (composite of service usage), `charge_ratio` (charges vs. contract-type peer average), and `support_intensity` (tickets per tenure month).

3. **Preprocessing** — `preprocessing.py` imputes missing values with column medians, one-hot encodes categoricals (contract type, internet service, payment method), standardizes numeric features to zero mean and unit variance, and persists the processed dataset to both SQLite and CSV. The fitted `StandardScaler` is saved separately for inference-time use.

4. **Training** — `train.py` splits data 80/20 with stratified sampling, then trains Logistic Regression (3 grid combos), Random Forest (12 combos), and XGBoost (24 combos) under 5-fold GridSearchCV optimizing ROC-AUC. Every model artifact and metadata JSON is saved to `models/saved/` and a timestamped copy to `models/versioned/`.

5. **Evaluation** — `evaluate.py` runs each trained model against the 20K held-out test set, computing accuracy, precision, recall, F1, and ROC-AUC. It selects the best model by ROC-AUC, generates confusion matrix PNGs and a combined ROC curve chart, and exports a `model_comparison.csv` + `model_comparison.json` report.

6. **Clustering** — `clustering.py` runs K-Means on six behavioral features using the original (unscaled) feature space. The elbow method and silhouette score analysis determine K=4. Each cluster is assigned a business label: *Loyal High-Value*, *At-Risk / Frustrated*, *Budget-Conscious*, and *New Power Users*.

7. **BI Output** — `bi_output.py` exports three Tableau-ready CSVs (`churn_analysis_tableau.csv`, `segment_summary.csv`, `executive_kpi.csv`), executive KPI JSON (total customers, churn rate, revenue at risk, model performance), and all visualization PNGs to `outputs/`.

8. **Inference API** — `app.py` loads `xgboost.joblib` and `scaler.joblib` at startup. `POST /predict-churn` accepts a Pydantic-validated `CustomerFeatures` payload, aligns features to the scaler's expected columns, transforms, and returns `churn_probability` (0–1), `risk_level` (High/Medium/Low), and `model_version`.

---

## 🧠 Key Techniques

- **Multi-model comparison under GridSearchCV** — all three models are tuned with the same 5-fold CV framework and ROC-AUC metric before selection, so the winner is objectively the best and not just the default choice
- **Stratified train-test split** — the churn ratio (typically 20–30%) is preserved in both train and test sets, preventing evaluation bias on an imbalanced label
- **Separate scaler persistence** — the `StandardScaler` is saved independently from the model because it must exactly match training-time scaling at inference; model and preprocessing artifacts have different lifecycle requirements
- **Model versioning with audit trail** — every training run saves a timestamped copy to `models/versioned/`, enabling rollback to any prior version without re-running the pipeline
- **Business-semantic feature engineering** — derived features (`support_intensity`, `charge_ratio`, `engagement_score`) encode domain knowledge that raw counts and totals don't express; they consistently rank in the top 10 feature importances
- **Behavioral segmentation on original feature space** — K-Means clusters on the unencoded, interpretable features so segment profiles map back directly to business attributes (tenure, spend, usage)
- **BI-first output design** — `bi_output.py` produces files designed to be drag-dropped into Tableau or Power BI with no additional transformation — column names, data types, and aggregation levels are chosen for analyst usability

---

## 📊 Results

### Model Performance (on 20K held-out test set)

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 84.5% | 0.9244 | 0.8417 | 0.8487 | 0.8452 |
| Random Forest | 84.0% | 0.9225 | 0.8375 | 0.8422 | 0.8398 |
| **XGBoost** ✓ | **85.1%** | **0.9328** | **0.8496** | **0.8523** | **0.8509** |

XGBoost is selected as the production model. Best CV score: **0.9322** (5-fold, ROC-AUC).

### Top Predictive Features (XGBoost)

| Rank | Feature | Importance | Business Signal |
|---|---|---|---|
| 1 | `contract_type_Two_year` | 0.152 | Long-term contracts dramatically reduce churn |
| 2 | `tenure_months` | 0.111 | Longer tenure = lower risk |
| 3 | `contract_type_One_year` | 0.107 | Annual contracts moderately reduce risk |
| 4 | `support_intensity` | 0.107 | Frequent tickets signal frustration |
| 5 | `monthly_charges` | 0.079 | Higher bills increase churn risk |
| 6 | `internet_service_Fiber_optic` | 0.077 | Fiber customers churn more |
| 7 | `charge_ratio` | 0.052 | Overpaying vs. peers increases risk |

### Customer Segments (K-Means, K=4)

| Segment | Profile | Recommended Action |
|---|---|---|
| **Loyal High-Value** | Long tenure, low support tickets, high engagement | Reward programs, early renewal incentives |
| **At-Risk / Frustrated** | High support intensity, moderate tenure | Proactive outreach, dedicated CSM, service audit |
| **Budget-Conscious** | Low monthly spend, month-to-month contracts | Value-add bundle offers, upgrade promotions |
| **New Power Users** | Short tenure, high usage and charges | Onboarding optimization, early commitment discounts |

---

## 💡 Business Impact

- **Quantifies revenue at risk** — the executive KPI output computes `at_risk_customers × avg_monthly_charges × 12`, giving leadership a dollar figure to size the retention investment against
- **Prioritizes the right customers** — the three-tier risk classification (High / Medium / Low) lets retention teams triage a 100K customer base down to the cohort that needs immediate action
- **Makes segmentation actionable** — four behavioral clusters with business-labeled profiles map directly to differentiated retention playbooks instead of generic, one-size-fits-all campaigns
- **Accelerates BI workflows** — Tableau-ready exports and pre-computed KPI summaries mean analysts skip data wrangling and go straight to dashboard building
- **Scales to production** — Docker containerization, a clean environment variable interface, and a Redshift-ready data layer mean the local SQLite setup can be promoted to a cloud data warehouse without code changes
- **Audit-ready by design** — versioned model artifacts, per-run metadata JSON, and a full model comparison report create a traceable lineage from training parameters to production predictions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd customer-churn-ml-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Run the Full Pipeline

```bash
python main.py
```

This single command:
1. Generates 100K synthetic customer records
2. Engineers features and preprocesses data
3. Trains Logistic Regression, Random Forest, and XGBoost under GridSearchCV
4. Evaluates all models and selects the best by ROC-AUC
5. Runs K-Means behavioral segmentation
6. Exports all BI outputs, visualizations, and executive KPIs
7. Prints an executive summary to stdout

### Start the Inference API

```bash
# Models must be trained first (run the pipeline above)
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Test a Prediction

```bash
curl -X POST http://localhost:8000/predict-churn \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure_months": 6,
    "monthly_charges": 85.0,
    "total_charges": 510.0,
    "num_support_tickets": 3,
    "monthly_minutes_used": 200,
    "data_usage_gb": 5.0,
    "num_dependents": 0,
    "contract_type_One_year": 0,
    "contract_type_Two_year": 0,
    "internet_service_Fiber_optic": 1,
    "internet_service_No": 0,
    "paperless_billing_Yes": 1
  }'
```

```json
{
  "churn_probability": 0.8342,
  "risk_level": "High",
  "model_version": "1.0.0"
}
```

### Run Tests

```bash
pytest tests/ -v
```

### Docker

```bash
# Run the inference API in a container
docker-compose up churn-api

# Run the training pipeline in a container
docker-compose --profile training run train
```

---

## 📁 Project Structure

```
customer-churn-ml-pipeline/
├── src/
│   ├── config/
│   │   └── settings.py              # Central path and config constants
│   ├── data/
│   │   ├── ingestion.py             # Synthetic data generation, CSV/S3 save
│   │   ├── feature_engineering.py  # Tenure buckets, engagement score, charge ratio
│   │   └── preprocessing.py        # Imputation, encoding, scaling, SQLite storage
│   ├── models/
│   │   ├── train.py                 # GridSearchCV training for all 3 models
│   │   ├── evaluate.py              # Metrics, ROC curves, confusion matrices
│   │   ├── predict.py               # Single and batch inference utilities
│   │   └── clustering.py            # K-Means segmentation, elbow method
│   ├── pipelines/
│   │   └── training_pipeline.py    # End-to-end orchestrator (called by main.py)
│   ├── api/
│   │   └── app.py                   # FastAPI inference server
│   └── utils/
│       ├── logger.py                # Centralized structured logging
│       ├── validation.py            # Data quality checks
│       └── bi_output.py             # Tableau CSVs, executive KPI exports
│
├── data/
│   ├── raw/                         # Generated raw CSV
│   ├── processed/                   # Cleaned data + SQLite DB
│   ├── external/                    # External data sources
│   └── s3_simulation/               # Simulated S3 bucket structure
│
├── models/
│   ├── saved/                       # Latest model artifacts (overwritten each run)
│   │   ├── xgboost.joblib
│   │   ├── random_forest.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── scaler.joblib
│   │   └── *_metadata.json
│   └── versioned/                   # Timestamped copies for audit and rollback
│
├── outputs/
│   ├── tableau/                     # churn_analysis_tableau.csv, segment_summary.csv
│   ├── reports/                     # model_comparison.csv/json, executive_kpi.csv/json
│   └── visualizations/              # ROC curves, confusion matrices, feature importance
│
├── docs/                            # Full documentation (see docs/INDEX.md)
├── tests/
│   └── test_pipeline.py
├── main.py                          # Pipeline entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🌐 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns status and loaded model version |
| `/model-info` | GET | Training metadata — params, CV score, version, timestamp |
| `/predict-churn` | POST | Predict churn probability and risk level for a single customer |

### POST /predict-churn — Input Schema

| Field | Type | Required | Description |
|---|---|---|---|
| `age` | int | Yes | Customer age (18–100) |
| `tenure_months` | int | Yes | Months as customer (0–72) |
| `monthly_charges` | float | Yes | Monthly bill amount |
| `total_charges` | float | Yes | Total charges to date |
| `num_support_tickets` | int | Yes | Number of support tickets filed |
| `monthly_minutes_used` | int | Yes | Monthly phone minutes |
| `data_usage_gb` | float | Yes | Monthly data usage in GB |
| `num_dependents` | int | Yes | Number of dependents |
| `contract_type_One_year` | int | No | 1 if on one-year contract, else 0 |
| `contract_type_Two_year` | int | No | 1 if on two-year contract, else 0 |
| `internet_service_Fiber_optic` | int | No | 1 if fiber optic internet, else 0 |
| `internet_service_No` | int | No | 1 if no internet service, else 0 |
| `paperless_billing_Yes` | int | No | 1 if paperless billing, else 0 |

**Note:** If both contract type flags are 0, the customer is on a month-to-month contract (the default reference category).

### POST /predict-churn — Response Schema

| Field | Type | Description |
|---|---|---|
| `churn_probability` | float | Churn probability 0.0–1.0, 4 decimal places |
| `risk_level` | string | `"High"` (≥0.70) · `"Medium"` (0.40–0.69) · `"Low"` (<0.40) |
| `model_version` | string | Model version in use (e.g., `"1.0.0"`) |

---

## 🔧 Configuration

```env
# AWS credentials (simulated — swap for real keys in production)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=churn-ml-pipeline

# Redshift (simulated — replace with real connection string for cloud deploy)
REDSHIFT_HOST=localhost
REDSHIFT_PORT=5439
REDSHIFT_DB=churn_analytics

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_VERSION=1.0.0

# Logging
LOG_LEVEL=INFO
```

To move to production: replace the SQLite path in `settings.py` with your Redshift connection string, and replace the S3 simulation directory with a real `boto3` client using the credentials above.

---

## 📚 Documentation

Full technical documentation is in the [`docs/`](docs/) folder:

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, component interactions, design decisions |
| [Data Pipeline](docs/data-pipeline.md) | Ingestion, feature engineering, preprocessing details |
| [ML Models](docs/ml-models.md) | Training process, hyperparameters, evaluation, feature importance |
| [Clustering & Segmentation](docs/clustering-segmentation.md) | K-Means methodology, elbow method, segment profiles |
| [API Reference](docs/api-reference.md) | Full endpoint docs, schemas, error codes, usage examples |
| [Deployment Guide](docs/deployment.md) | Docker, local setup, AWS cloud deployment |
| [Configuration Guide](docs/configuration.md) | Environment variables, settings, customization |
| [BI & Tableau Guide](docs/bi-outputs.md) | Tableau datasets, executive KPIs, visualization guide |

---

## 📌 Key Takeaways

- Demonstrates end-to-end ML engineering — data generation, feature engineering, multi-model training, automated evaluation, clustering, BI export, and a served REST API working together in a single coherent pipeline
- Applied **production ML engineering practices**: stratified splitting, GridSearchCV with cross-validation, separate scaler persistence, versioned model artifacts, and Pydantic-validated inference endpoints
- Feature engineering is **domain-driven** — derived features like `support_intensity` and `charge_ratio` encode business logic that consistently outperforms raw counts in feature importance
- **BI-first output philosophy** — every export is designed for direct analyst consumption, not as an intermediate artifact that requires additional transformation before it can be used
- Clean separation of concerns across ingestion, engineering, modeling, clustering, API, and output layers makes each stage independently testable and replaceable without touching the rest of the pipeline

---

## 📝 License

MIT
