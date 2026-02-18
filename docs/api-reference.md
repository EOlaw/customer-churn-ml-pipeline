# API Reference

This document covers the FastAPI inference endpoint — all available routes, request/response schemas, authentication, error handling, and usage examples.

---

## Overview

The API serves the trained XGBoost model for real-time churn predictions. It loads the model and scaler at startup and provides a REST interface for single-customer predictions.

**File:** `src/api/app.py`

**Base URL:** `http://localhost:8000`

**Auto-generated docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Starting the API

### Local Development

```bash
# First, train models (required before API can serve predictions)
python main.py

# Start the API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up churn-api
```

### What Happens at Startup

1. FastAPI initializes the application
2. The `load_models()` startup event fires
3. `models/saved/xgboost.joblib` is loaded into memory
4. `models/saved/scaler.joblib` is loaded into memory
5. Feature names are extracted from the scaler for alignment
6. The API is ready to serve predictions

If models are not found (pipeline hasn't been run), the API starts but returns 503 errors on prediction requests.

---

## Endpoints

### GET /health

Health check endpoint for monitoring and load balancers.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response (model loaded):**
```json
{
  "status": "healthy",
  "model_version": "1.0.0"
}
```

**Response (model not loaded):**
```json
{
  "status": "model_not_loaded",
  "model_version": "1.0.0"
}
```

---

### POST /predict-churn

Predict churn probability for a single customer.

**Request:**
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

**Response:**
```json
{
  "churn_probability": 0.8342,
  "risk_level": "High",
  "model_version": "1.0.0"
}
```

#### Input Schema: CustomerFeatures

| Field | Type | Required | Range | Default | Description |
|-------|------|----------|-------|---------|-------------|
| `age` | int | Yes | 18-100 | - | Customer age |
| `tenure_months` | int | Yes | 0-72 | - | Months as customer |
| `monthly_charges` | float | Yes | >= 0 | - | Monthly bill amount |
| `total_charges` | float | Yes | >= 0 | - | Total charges to date |
| `num_support_tickets` | int | Yes | >= 0 | - | Number of support tickets |
| `monthly_minutes_used` | int | Yes | >= 0 | - | Monthly phone minutes |
| `data_usage_gb` | float | Yes | >= 0 | - | Monthly data usage in GB |
| `num_dependents` | int | Yes | >= 0 | - | Number of dependents |
| `contract_type_One_year` | int | No | 0 or 1 | 0 | Is on one-year contract? |
| `contract_type_Two_year` | int | No | 0 or 1 | 0 | Is on two-year contract? |
| `internet_service_Fiber_optic` | int | No | 0 or 1 | 0 | Has fiber optic? |
| `internet_service_No` | int | No | 0 or 1 | 0 | No internet service? |
| `paperless_billing_Yes` | int | No | 0 or 1 | 0 | Uses paperless billing? |

**Note on contract type:** If both `contract_type_One_year` and `contract_type_Two_year` are 0, the customer is on a month-to-month contract (the default/reference category from one-hot encoding).

#### Output Schema: PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `churn_probability` | float | Probability of churn (0.0 to 1.0), rounded to 4 decimal places |
| `risk_level` | string | "High" (>=0.7), "Medium" (0.4-0.69), "Low" (<0.4) |
| `model_version` | string | Version of the model used (e.g., "1.0.0") |

---

### GET /model-info

Returns metadata about the currently loaded model.

**Request:**
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_name": "xgboost",
  "best_cv_score": 0.9322,
  "best_params": {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 200,
    "subsample": 1.0
  },
  "version": "1.0.0",
  "timestamp": "20260217_220843",
  "model_path": "/app/models/saved/xgboost.joblib"
}
```

---

## Error Handling

### 503 Service Unavailable

Returned when the model hasn't been loaded (pipeline hasn't been run):

```json
{
  "detail": "Model not loaded. Run training pipeline first."
}
```

**Fix:** Run `python main.py` to train models before starting the API.

### 422 Validation Error

Returned when the request body fails Pydantic validation:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 18",
      "input": 10
    }
  ]
}
```

**Common causes:**
- Missing required fields
- Values outside allowed ranges (e.g., age < 18)
- Wrong data types (e.g., string instead of int)

### 500 Internal Server Error

Returned if prediction fails due to feature mismatch or model corruption:

```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Usage Examples

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict-churn",
    json={
        "age": 28,
        "tenure_months": 3,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "num_support_tickets": 5,
        "monthly_minutes_used": 150,
        "data_usage_gb": 3.0,
        "num_dependents": 0,
        "contract_type_One_year": 0,
        "contract_type_Two_year": 0,
        "internet_service_Fiber_optic": 1,
        "internet_service_No": 0,
        "paperless_billing_Yes": 1,
    },
)

data = response.json()
print(f"Churn Probability: {data['churn_probability']:.1%}")
print(f"Risk Level: {data['risk_level']}")
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/predict-churn", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    age: 45,
    tenure_months: 48,
    monthly_charges: 55.0,
    total_charges: 2640.0,
    num_support_tickets: 1,
    monthly_minutes_used: 500,
    data_usage_gb: 8.0,
    num_dependents: 2,
    contract_type_One_year: 0,
    contract_type_Two_year: 1,
    internet_service_Fiber_optic: 0,
    internet_service_No: 0,
    paperless_billing_Yes: 0,
  }),
});

const data = await response.json();
console.log(`Risk: ${data.risk_level} (${data.churn_probability})`);
```

### Scenario Examples

**High-Risk Customer:**
```json
{
  "age": 25, "tenure_months": 2, "monthly_charges": 110.0,
  "total_charges": 220.0, "num_support_tickets": 4,
  "monthly_minutes_used": 100, "data_usage_gb": 2.0,
  "num_dependents": 0,
  "contract_type_One_year": 0, "contract_type_Two_year": 0,
  "internet_service_Fiber_optic": 1, "internet_service_No": 0,
  "paperless_billing_Yes": 1
}
→ churn_probability: ~0.90, risk_level: "High"
```
*Why:* Month-to-month, very short tenure, high charges, many support tickets, fiber optic, low usage.

**Low-Risk Customer:**
```json
{
  "age": 55, "tenure_months": 60, "monthly_charges": 45.0,
  "total_charges": 2700.0, "num_support_tickets": 0,
  "monthly_minutes_used": 600, "data_usage_gb": 10.0,
  "num_dependents": 3,
  "contract_type_One_year": 0, "contract_type_Two_year": 1,
  "internet_service_Fiber_optic": 0, "internet_service_No": 0,
  "paperless_billing_Yes": 0
}
→ churn_probability: ~0.10, risk_level: "Low"
```
*Why:* Two-year contract, 5-year tenure, moderate charges, no support issues, high usage, dependents.

---

## API Architecture

```
HTTP Request
    │
    ▼
FastAPI Router (/predict-churn)
    │
    ▼
Pydantic Validation (CustomerFeatures)
    │ Rejects invalid input (422)
    ▼
Feature Alignment
    │ Ensures columns match model's expectations
    ▼
StandardScaler Transform
    │ Scales numeric features using saved scaler
    ▼
XGBoost predict_proba()
    │ Returns [P(no churn), P(churn)]
    ▼
Risk Classification
    │ Maps probability to High/Medium/Low
    ▼
PredictionResponse (JSON)
```
