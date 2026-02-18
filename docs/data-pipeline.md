# Data Pipeline Documentation

This document covers the complete ETL (Extract, Transform, Load) pipeline — from synthetic data generation through feature engineering, preprocessing, and storage.

---

## Overview

The data pipeline transforms raw customer data into model-ready features through three stages:

```
Raw Data (20 columns)
    │
    ▼ feature_engineering.py
Engineered Data (25 columns)  ← 5 new business features added
    │
    ▼ preprocessing.py
Model-Ready Data (36 columns) ← Encoded + Scaled
    │
    ├──► SQLite Database (data/processed/churn_data.db)
    ├──► Processed CSV   (data/processed/customer_churn_processed.csv)
    └──► S3 Simulation   (data/s3_simulation/)
```

---

## Stage 1: Data Ingestion

**File:** `src/data/ingestion.py`

### What It Does

Generates a synthetic dataset of 100,000+ customer records that mimics real-world telecom/subscription service data. The data includes realistic correlations between features and churn outcomes.

### Functions

#### `generate_synthetic_data(n_samples=100000, seed=42)`

Creates the full dataset with the following feature categories:

**Demographics:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `customer_id` | String | CUST-000001 to CUST-100000 | Unique identifier |
| `gender` | Categorical | Male, Female | Customer gender |
| `age` | Integer | 18-85 | Normal distribution, mean=45, std=15 |
| `partner` | Categorical | Yes, No | Has a partner (48% / 52%) |
| `num_dependents` | Integer | 0-4 | Number of dependents |

**Account Information:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `tenure_months` | Integer | 1-72 | Months as customer (exponential dist, mean=30) |
| `contract_type` | Categorical | Month-to-month, One year, Two year | Contract term (50% / 25% / 25%) |

**Billing:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `monthly_charges` | Float | $18-$120 | Monthly bill amount |
| `total_charges` | Float | Varies | `monthly_charges * tenure * noise(0.85-1.15)` |
| `payment_method` | Categorical | 4 types | Electronic check (35%), Mailed check (20%), Bank transfer (22%), Credit card (23%) |
| `paperless_billing` | Categorical | Yes, No | Uses paperless billing (60% / 40%) |

**Services:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `internet_service` | Categorical | DSL, Fiber optic, No | Internet type (35% / 45% / 20%) |
| `online_security` | Categorical | Yes, No, No internet service | Has online security |
| `tech_support` | Categorical | Yes, No, No internet service | Has tech support |
| `streaming_tv` | Categorical | Yes, No, No internet service | Has streaming TV |
| `streaming_movies` | Categorical | Yes, No, No internet service | Has streaming movies |

**Usage Metrics:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `num_support_tickets` | Integer | 0+ | Poisson distribution, mean=1.5 |
| `monthly_minutes_used` | Integer | 0-1500 | Normal distribution, mean=450, std=150 |
| `data_usage_gb` | Float | 0+ | Exponential distribution, mean=8 GB |

### How Churn Labels Are Generated

The churn label is **not random** — it's generated using a weighted scoring algorithm that creates learnable patterns:

```python
churn_score = 0

# Contract type (strongest signal)
+3.0  if Month-to-month
+0.5  if One year
-2.0  if Two year

# Tenure
+2.5  if < 6 months
+1.0  if < 12 months
-1.5  if > 36 months
-1.0  if > 48 months

# Monthly charges
+1.5  if > $80
+1.0  if > $100
-1.0  if < $40

# Payment method
+1.2  if Electronic check

# Internet service
+1.0  if Fiber optic
-0.5  if No internet

# Service adoption
+0.8  if no online security
+0.6  if no tech support

# Usage
+0.6  per support ticket
+1.0  if minutes < 200
-0.5  if minutes > 600

# Demographics
+0.5  if no partner
+0.5  if 0 dependents

# Noise
+ Normal(0, 0.3)  ← minimal noise for clean signal
```

The score is converted to a churn probability using a **sigmoid function**:

```
churn_probability = 1 / (1 + exp(-1.0 * (score - median(score))))
```

The sigmoid centers around the median score, producing approximately 50% churn rate with clear separation between high-risk and low-risk customers.

### Missing Value Injection

Approximately 3% of values in `monthly_charges`, `total_charges`, and `data_usage_gb` are replaced with `NaN` to simulate real-world data quality issues.

#### `save_to_csv(df, filename="customer_churn_raw.csv")`

Saves the generated DataFrame to `data/raw/customer_churn_raw.csv`.

#### `simulate_s3_upload(source_path, s3_prefix="raw")`

Copies the CSV file to `data/s3_simulation/raw/` to simulate an AWS S3 bucket upload. In production, this would be replaced with actual `boto3` S3 operations.

#### `load_from_csv(filepath)`

Utility function to reload a saved CSV file back into a DataFrame.

#### `ingest_data()`

Orchestrates the full ingestion pipeline: generate → save → simulate S3 upload. Returns the DataFrame.

---

## Stage 2: Feature Engineering

**File:** `src/data/feature_engineering.py`

### What It Does

Creates 5 new derived features that capture business logic not directly expressed in the raw data. These features are engineered **before** encoding and scaling so they can be computed on the original values.

### Why Feature Engineering Matters

Raw data often contains the "what" but not the "so what." For example:
- Raw: "Customer has 4 support tickets and 24 months of tenure"
- Engineered: "Customer has 0.17 support tickets per month" (support_intensity)

The engineered version tells us whether 4 tickets is concerning (new customer) or normal (long-term customer).

### Functions

#### `create_tenure_buckets(df) → DataFrame`

**What:** Converts continuous `tenure_months` into categorical lifecycle stages.

**Buckets:**
| Bucket | Range | Business Meaning |
|--------|-------|-----------------|
| `0-6mo` | 0-6 months | New customer, high churn risk |
| `6-12mo` | 6-12 months | Early relationship, still deciding |
| `1-2yr` | 12-24 months | Establishing loyalty |
| `2-4yr` | 24-48 months | Established customer |
| `4-6yr` | 48-72 months | Long-term loyal customer |

**Why:** Churn risk is not linear with tenure — it drops sharply after the first year, then stabilizes. Buckets capture this non-linearity.

#### `create_avg_monthly_spend(df) → DataFrame`

**What:** Calculates `total_charges / tenure_months`.

**Formula:**
```
avg_monthly_spend = total_charges / tenure_months   (if tenure > 0)
avg_monthly_spend = monthly_charges                 (if tenure = 0)
```

**Why:** `total_charges` alone doesn't tell you if a customer is a high or low spender — a customer with $5,000 total charges over 60 months spends less per month than one with $2,000 over 12 months.

#### `create_engagement_score(df) → DataFrame`

**What:** A composite score (0 to 1) measuring how engaged a customer is with the service.

**Formula:**
```
engagement_score = 0.3 * (minutes / max_minutes)
                 + 0.3 * (data_gb / max_data_gb)
                 + 0.4 * (services_adopted / total_services)
```

**Components:**
- **Minutes usage** (30% weight): How much they use the phone service
- **Data usage** (30% weight): How much data they consume
- **Service adoption** (40% weight): How many services they subscribe to (online_security, tech_support, streaming_tv, streaming_movies)

**Why:** Engaged customers are less likely to churn. This single metric combines multiple engagement signals.

#### `create_charge_ratio(df) → DataFrame`

**What:** Ratio of a customer's monthly charges to the average for their contract type.

**Formula:**
```
charge_ratio = monthly_charges / avg_monthly_charges_for_contract_type
```

**Interpretation:**
- `charge_ratio > 1.0`: Paying more than average for their contract type
- `charge_ratio < 1.0`: Paying less than average
- `charge_ratio ≈ 1.0`: Average spending

**Why:** A customer paying $100/month on a month-to-month contract might be normal, but $100/month on a two-year contract could indicate they're overpaying — increasing churn risk.

#### `create_support_intensity(df) → DataFrame`

**What:** Support tickets normalized by tenure.

**Formula:**
```
support_intensity = num_support_tickets / tenure_months
```

**Why:** 5 support tickets over 5 years is normal (0.08/month). 5 tickets in 2 months is a red flag (2.5/month). Raw ticket count doesn't capture urgency.

#### `engineer_features(df) → DataFrame`

Orchestrator function that applies all five transformations sequentially. Logs each step and returns the enhanced DataFrame (25 columns).

---

## Stage 3: Preprocessing

**File:** `src/data/preprocessing.py`

### What It Does

Transforms the engineered dataset into a format that machine learning algorithms can consume: no missing values, all numeric, standardized scale.

### Functions

#### `handle_missing_values(df) → DataFrame`

**Strategy:** Median imputation for all numeric columns.

**Why median (not mean)?**
- Median is robust to outliers
- Monthly charges and total charges have right-skewed distributions
- Mean imputation would be pulled toward high outliers

**Columns affected:**
- `monthly_charges` (~3,063 missing values, median ≈ $68.91)
- `total_charges` (~3,054 missing values, median ≈ $1,212.79)
- `data_usage_gb` (~3,033 missing values, median ≈ 5.53 GB)
- Derived features (`avg_monthly_spend`, `engagement_score`, `charge_ratio`) also have missing values where their source columns had NaN

#### `encode_categorical(df) → (DataFrame, dict)`

**Strategy:** One-hot encoding with `drop_first=True`.

**What `drop_first=True` means:**
A categorical column with 3 values (A, B, C) becomes 2 binary columns (B=0/1, C=0/1). If both are 0, it's A. This avoids the **dummy variable trap** where perfect multicollinearity breaks linear models.

**Columns encoded (11 total):**
| Original Column | Values | Encoded Columns Created |
|-----------------|--------|------------------------|
| `gender` | Male, Female | `gender_Male` |
| `partner` | Yes, No | `partner_Yes` |
| `contract_type` | 3 values | `contract_type_One year`, `contract_type_Two year` |
| `payment_method` | 4 values | 3 columns |
| `paperless_billing` | Yes, No | `paperless_billing_Yes` |
| `internet_service` | 3 values | 2 columns |
| `online_security` | 3 values | 2 columns |
| `tech_support` | 3 values | 2 columns |
| `streaming_tv` | 3 values | 2 columns |
| `streaming_movies` | 3 values | 2 columns |
| `tenure_bucket` | 5 values | 4 columns |

**Result:** 20 raw columns → 36 total columns (including customer_id and churn)

**Returns:** The encoded DataFrame and a dictionary mapping each original column to its unique values (for reference/debugging).

#### `scale_features(df, target_col="churn") → (DataFrame, StandardScaler)`

**Strategy:** StandardScaler (z-score normalization).

**Formula for each feature:**
```
scaled_value = (value - mean) / standard_deviation
```

**After scaling:**
- Mean of each feature = 0
- Standard deviation = 1

**Why scale?**
- Logistic Regression uses gradient descent — features on different scales slow convergence
- Distance-based algorithms (K-Means) need equal feature weighting
- XGBoost and Random Forest don't strictly need scaling, but it doesn't hurt

**Excluded from scaling:**
- `churn` (target variable — must remain 0/1)
- `customer_id` (identifier, not a feature)

**Returns:** Scaled DataFrame and the fitted `StandardScaler` object (saved for inference).

#### `store_to_sql(df, table_name="customer_data")`

Stores the fully processed DataFrame into SQLite at `data/processed/churn_data.db`. The table is replaced on each run (`if_exists="replace"`).

**How to query the database:**
```python
import sqlite3
conn = sqlite3.connect("data/processed/churn_data.db")
df = pd.read_sql("SELECT * FROM customer_data WHERE churn = 1 LIMIT 10", conn)
```

#### `export_processed_csv(df, filename="customer_churn_processed.csv")`

Exports the processed data to `data/processed/customer_churn_processed.csv` for external tools.

#### `preprocess(df) → (DataFrame, StandardScaler, dict)`

Orchestrates the full preprocessing pipeline:
1. Validate raw data
2. Handle missing values
3. Encode categoricals
4. Scale features
5. Validate processed data
6. Store to SQL
7. Export CSV

Returns the processed DataFrame, the fitted scaler, and the encoding map.

---

## Data Validation

**File:** `src/utils/validation.py`

### What It Does

Quality gates that run before and after preprocessing to catch data issues early.

### Raw Data Validation (`validate_raw_data`)

Checks:
| Check | What It Catches |
|-------|----------------|
| Missing columns | Schema drift, corrupted data |
| Empty DataFrame | Failed data load |
| Duplicate customer_ids | Data duplication bugs |
| Negative numeric values | Invalid age, charges, etc. |

### Processed Data Validation (`validate_processed_data`)

Checks:
| Check | What It Catches |
|-------|----------------|
| Remaining null values | Incomplete imputation |

### Required Columns (20)

```
customer_id, gender, age, tenure_months, contract_type,
monthly_charges, total_charges, payment_method,
num_support_tickets, internet_service, online_security,
tech_support, streaming_tv, streaming_movies,
paperless_billing, num_dependents, partner,
monthly_minutes_used, data_usage_gb, churn
```

---

## S3 Simulation

The system simulates AWS S3 using a local directory structure:

```
data/s3_simulation/
├── raw/
│   └── customer_churn_raw.csv     ← simulates s3://churn-ml-pipeline/raw/
├── processed/                      ← simulates s3://churn-ml-pipeline/processed/
└── models/                         ← simulates s3://churn-ml-pipeline/models/
```

**In production:** Replace `simulate_s3_upload()` with:
```python
import boto3
s3 = boto3.client('s3')
s3.upload_file(source_path, S3_BUCKET_NAME, f"{prefix}/{filename}")
```

The AWS credentials are configured via environment variables in `.env`.

---

## Data Storage Summary

| Storage | Location | Format | Purpose |
|---------|----------|--------|---------|
| Raw CSV | `data/raw/customer_churn_raw.csv` | CSV, 100K rows, 20 columns | Original generated data |
| S3 Simulation | `data/s3_simulation/raw/` | CSV copy | Simulated cloud storage |
| SQLite DB | `data/processed/churn_data.db` | SQLite table `customer_data` | SQL-queryable processed data |
| Processed CSV | `data/processed/customer_churn_processed.csv` | CSV, 100K rows, 36 columns | Model-ready dataset |
