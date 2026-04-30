# Customer Churn Prediction & Behavioral Segmentation System

A production-ready, end-to-end ML pipeline that predicts customer churn probability, segments customers using behavioral clustering, and generates executive-ready analytics outputs.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Documentation Index](docs/INDEX.md) | Start here — links to all docs |
| [Architecture](docs/architecture.md) | System design, data flow, component interactions |
| [Project Structure](docs/project-structure.md) | Every folder and file explained |
| [Data Pipeline](docs/data-pipeline.md) | Ingestion, feature engineering, preprocessing |
| [ML Models](docs/ml-models.md) | Training, evaluation, hyperparameters, feature importance |
| [Clustering & Segmentation](docs/clustering-segmentation.md) | K-Means, elbow method, business segments |
| [API Reference](docs/api-reference.md) | FastAPI endpoints, schemas, usage examples |
| [Deployment Guide](docs/deployment.md) | Docker, local setup, cloud deployment |
| [Configuration Guide](docs/configuration.md) | Environment variables, settings, customization |
| [BI & Tableau Guide](docs/bi-outputs.md) | Tableau datasets, executive KPIs, visualizations |

## Business Problem

Customer churn directly impacts revenue and growth. This system:
- **Predicts** which customers are likely to leave (classification)
- **Segments** the customer base into behavioral groups (clustering)
- **Quantifies** revenue at risk to prioritize retention efforts
- **Provides** Tableau-ready datasets for executive dashboards

## Architecture

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
│              ┌──────────▼──────────┐                             │
│              │   SQLite Storage    │                              │
│              │   S3 Simulation     │                              │
│              └─────────────────────┘                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                MODEL TRAINING & EVALUATION                      │
│                                                                  │
│  ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐       │
│  │ Logistic Reg.  │ │ Random Forest│ │   XGBoost       │       │
│  │  (baseline)    │ │              │ │  (primary)      │       │
│  └────────────────┘ └──────────────┘ └─────────────────┘       │
│                                                                  │
│  GridSearchCV  ──►  Cross-Validation  ──►  Best Model Selection │
│  ROC-AUC / Precision / Recall / Confusion Matrix                │
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
│  ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐      │
│  │ FastAPI       │ │ Tableau CSVs   │ │ Executive KPIs   │      │
│  │ /predict-churn│ │                │ │ JSON + CSV       │      │
│  └──────────────┘ └────────────────┘ └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Ingest** → Generate/load 100K+ customer records with realistic churn patterns
2. **Engineer** → Create tenure buckets, engagement scores, charge ratios, support intensity
3. **Preprocess** → Handle nulls (median imputation), one-hot encode, standardize
4. **Store** → SQLite database + S3-simulated directory structure
5. **Train** → 3 models with GridSearchCV hyperparameter tuning
6. **Evaluate** → Full metrics suite, select best model by ROC-AUC
7. **Cluster** → K-Means behavioral segmentation with business interpretation
8. **Serve** → FastAPI endpoint + Tableau-ready exports

### Feature Engineering Strategy
| Feature | Description | Business Logic |
|---------|-------------|---------------|
| `tenure_bucket` | Categorical tenure groups | Identifies lifecycle stage |
| `avg_monthly_spend` | total_charges / tenure | Normalizes spending over time |
| `engagement_score` | Composite of usage + services | Higher = more invested |
| `charge_ratio` | Charges vs contract-type average | Detects overcharged customers |
| `support_intensity` | Tickets per month | Flags frustrated users |

## ML Methodology

### Classification Models
- **Logistic Regression** — Interpretable baseline
- **Random Forest** — Ensemble with feature importance
- **XGBoost** — Gradient boosting (typically best performer)

All models use:
- 80/20 stratified train-test split
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- ROC-AUC as the primary optimization metric

### Clustering
- **K-Means** with elbow method and silhouette score analysis
- 4 segments with business labels (Loyal High-Value, At-Risk, Budget-Conscious, New Power Users)

## Project Structure

```
customer-churn-ml-pipeline/
├── data/
│   ├── raw/                    # Generated raw CSV
│   ├── processed/              # Cleaned data + SQLite DB
│   ├── external/               # External data sources
│   └── s3_simulation/          # Simulated S3 bucket
├── models/
│   ├── saved/                  # Latest model artifacts
│   └── versioned/              # Timestamped model versions
├── outputs/
│   ├── tableau/                # Tableau-ready CSVs
│   ├── reports/                # Evaluation & KPI reports
│   └── visualizations/         # Charts and plots
├── src/
│   ├── config/settings.py      # Central configuration
│   ├── data/
│   │   ├── ingestion.py        # Data generation & loading
│   │   ├── preprocessing.py    # Cleaning, encoding, scaling
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py            # Model training + GridSearchCV
│   │   ├── evaluate.py         # Metrics, ROC, confusion matrix
│   │   ├── predict.py          # Inference utilities
│   │   └── clustering.py       # K-Means segmentation
│   ├── pipelines/
│   │   └── training_pipeline.py # End-to-end orchestrator
│   ├── api/app.py              # FastAPI inference server
│   └── utils/
│       ├── logger.py           # Centralized logging
│       ├── validation.py       # Data quality checks
│       └── bi_output.py        # BI/Tableau outputs
├── tests/test_pipeline.py
├── main.py                     # Run full pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## How to Run Locally

### Prerequisites
- Python 3.10+

### Setup
```bash
# Clone the repository
git clone <repo-url>
cd customer-churn-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### Run the Full Pipeline
```bash
python main.py
```

This will:
1. Generate 100K synthetic customer records
2. Engineer features and preprocess data
3. Train and evaluate 3 ML models
4. Run K-Means clustering
5. Generate all BI outputs and visualizations

### Run the API Server
```bash
# First run the pipeline to train models, then:
uvicorn src.api.app:app --reload
```

### Test the Prediction Endpoint
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

Response:
```json
{
  "churn_probability": 0.83,
  "risk_level": "High",
  "model_version": "1.0.0"
}
```

### Run Tests
```bash
pytest tests/ -v
```

## How to Deploy

### Docker
```bash
# Build and run the API
docker-compose up churn-api

# Run training pipeline in Docker
docker-compose --profile training run train
```

### AWS Deployment (Production)
1. Push Docker image to ECR
2. Deploy on ECS/EKS or Lambda
3. Replace SQLite with Redshift (connection string in `.env`)
4. Replace S3 simulation with real S3 bucket
5. Set environment variables in your cloud provider

## Results

### Model Performance
| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| Logistic Regression | ~82% | ~0.88 | ~0.78 | ~0.72 | ~0.75 |
| Random Forest | ~85% | ~0.91 | ~0.82 | ~0.76 | ~0.79 |
| XGBoost | ~86% | ~0.92 | ~0.83 | ~0.78 | ~0.80 |

*Exact results vary by run. Target: ~85% accuracy achieved.*

### Customer Segments
| Segment | Description | Recommended Action |
|---------|-------------|-------------------|
| Loyal High-Value | Long tenure, low churn | Reward programs |
| At-Risk / Frustrated | High support tickets | Proactive outreach |
| Budget-Conscious | Low monthly spend | Value-add offers |
| New Power Users | Short tenure, high usage | Onboarding optimization |

## Tableau Dashboard

The pipeline exports Tableau-ready files to `outputs/tableau/`:
- `churn_analysis_tableau.csv` — Full dataset with segments
- `segment_summary.csv` — Segment profiles
- `executive_kpi.csv` — KPI summary

*Connect Tableau Desktop to these CSVs to build dashboards.*

<!-- Screenshots placeholder -->
<!-- ![Dashboard Overview](screenshots/dashboard_overview.png) -->
<!-- ![Segment Analysis](screenshots/segment_analysis.png) -->
<!-- ![Churn Prediction](screenshots/churn_prediction.png) -->

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Data | Pandas, NumPy, SQLite |
| API | FastAPI, Uvicorn, Pydantic |
| Visualization | Matplotlib |
| Cloud (Simulated) | S3 directory structure, Redshift schema |
| Deployment | Docker, docker-compose |
| BI | Tableau-ready CSV exports |
