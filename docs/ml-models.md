# Machine Learning Models Documentation

This document covers the complete ML methodology — model selection rationale, training process, hyperparameter tuning, evaluation metrics, feature importance analysis, and how to use the trained models.

---

## Overview

The system trains three classification models to predict customer churn (binary: 0 = stays, 1 = leaves). Each model is tuned using GridSearchCV with 5-fold cross-validation, and the best model is selected by ROC-AUC score.

```
Processed Data (36 features)
    │
    ├──► 80% Training Set (stratified)
    │         │
    │         ▼
    │    GridSearchCV (5-fold CV)
    │    ├── Logistic Regression
    │    ├── Random Forest
    │    └── XGBoost
    │         │
    │         ▼
    │    Best hyperparameters selected
    │    Models saved to models/saved/
    │
    └──► 20% Test Set (held out)
              │
              ▼
         Evaluation on unseen data
         ├── Accuracy, Precision, Recall, F1
         ├── ROC-AUC
         ├── Confusion Matrices
         └── Best model selected
```

---

## Model Training

**File:** `src/models/train.py`

### Data Splitting

```python
split_data(df, target_col="churn")
```

- **Split ratio:** 80% train / 20% test
- **Stratification:** The churn ratio is preserved in both splits
- **Random seed:** 42 (reproducible)
- **Excluded columns:** `customer_id` (identifier) and `churn` (target)
- **Result:** 80,000 training samples, 20,000 test samples

### Model 1: Logistic Regression (Baseline)

**What it is:** A linear model that predicts the probability of churn using a logistic (sigmoid) function. It learns a weight for each feature and sums them to produce a log-odds score.

**Why it's included:**
- Serves as a **baseline** — if more complex models can't beat it, the problem may not need complexity
- Highly **interpretable** — coefficients directly show feature impact
- Very **fast** to train
- Surprisingly effective when relationships are approximately linear

**Hyperparameter search space:**

| Parameter | Values Tested | Meaning |
|-----------|--------------|---------|
| `C` | 0.01, 0.1, 1.0 | Inverse regularization strength. Smaller = more regularization (simpler model) |

**Other settings:**
- `max_iter=1000`: Maximum iterations for convergence
- `solver=lbfgs`: Default solver, works well for L2 regularization
- Total grid combinations: 3

**How to interpret coefficients:**
```python
import joblib
model = joblib.load("models/saved/logistic_regression.joblib")
# Positive coefficient = increases churn probability
# Negative coefficient = decreases churn probability
# Magnitude = strength of effect
```

### Model 2: Random Forest

**What it is:** An ensemble of decision trees. Each tree is trained on a random subset of data and features. The final prediction is the majority vote across all trees.

**Why it's included:**
- Handles **non-linear relationships** naturally
- Resistant to **overfitting** (averaging many trees)
- Provides **feature importance** based on impurity reduction
- No feature scaling required (but we scale anyway for consistency)

**Hyperparameter search space:**

| Parameter | Values Tested | Meaning |
|-----------|--------------|---------|
| `n_estimators` | 100, 200 | Number of trees in the forest |
| `max_depth` | 10, 20, None | Maximum depth of each tree. `None` = no limit |
| `min_samples_split` | 2, 5 | Minimum samples required to split a node |

**Other settings:**
- `n_jobs=-1`: Use all CPU cores
- `random_state=42`: Reproducible
- Total grid combinations: 12

**How feature importance works:**
Each time a feature is used to split a node, the impurity (Gini) decreases. The total decrease across all trees is normalized to sum to 1.0, giving the importance score.

### Model 3: XGBoost (Primary)

**What it is:** A gradient boosting algorithm that builds trees sequentially. Each new tree corrects the errors of the previous ones. Uses second-order gradients for optimization.

**Why it's included:**
- Typically the **best-performing** model for tabular data
- Built-in **regularization** prevents overfitting
- Handles **missing values** natively (though we impute them)
- Excellent **feature importance** via gain, weight, or cover

**Hyperparameter search space:**

| Parameter | Values Tested | Meaning |
|-----------|--------------|---------|
| `n_estimators` | 100, 200 | Number of boosting rounds |
| `max_depth` | 3, 6, 9 | Maximum depth of each tree (3 = shallow, conservative) |
| `learning_rate` | 0.01, 0.1 | Step size shrinkage. Lower = more conservative, needs more trees |
| `subsample` | 0.8, 1.0 | Fraction of samples used per tree. 0.8 = stochastic boosting |

**Other settings:**
- `eval_metric="logloss"`: Binary cross-entropy loss
- `n_jobs=-1`: Use all CPU cores
- Total grid combinations: 24

**How XGBoost differs from Random Forest:**
| Aspect | Random Forest | XGBoost |
|--------|--------------|---------|
| Tree building | Independent (parallel) | Sequential (corrective) |
| Overfitting | Averaging reduces variance | Regularization + learning rate |
| Speed | Fast (parallelizable) | Slower per tree, but fewer trees needed |
| Missing values | Requires imputation | Handles natively |

---

## Hyperparameter Tuning

### GridSearchCV Explained

GridSearchCV exhaustively tries every combination of hyperparameters using cross-validation:

1. **Grid:** All combinations of parameter values (e.g., 24 for XGBoost)
2. **Cross-validation:** Each combination is evaluated using 5-fold CV
3. **Scoring:** ROC-AUC is the optimization metric
4. **Total fits:** Grid size × 5 folds (e.g., 24 × 5 = 120 for XGBoost)

```
5-Fold Cross-Validation:

Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]

Average score across 5 folds → CV score for this parameter combination
```

### Why ROC-AUC as the Metric?

- **Accuracy** can be misleading with imbalanced classes (e.g., 95% accuracy by always predicting "no churn")
- **ROC-AUC** measures the model's ability to **distinguish** between churners and non-churners across all threshold values
- A score of 0.5 = random guessing, 1.0 = perfect separation

---

## Model Evaluation

**File:** `src/models/evaluate.py`

### Metrics Computed

For each model, the following metrics are computed on the **held-out test set** (20,000 samples):

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / Total |
| **Precision** | Of predicted churners, how many actually churned? | TP / (TP + FP) |
| **Recall** | Of actual churners, how many did we catch? | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall | 2 × (Precision × Recall) / (Precision + Recall) |
| **ROC-AUC** | Area under the ROC curve | Probability model ranks a random positive higher than negative |

**Where:**
- TP = True Positive (predicted churn, actually churned)
- TN = True Negative (predicted stay, actually stayed)
- FP = False Positive (predicted churn, actually stayed) — wastes retention budget
- FN = False Negative (predicted stay, actually churned) — missed opportunity

### Achieved Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| Logistic Regression | 84.5% | 0.9244 | 0.8417 | 0.8487 | 0.8452 |
| Random Forest | 84.0% | 0.9225 | 0.8375 | 0.8422 | 0.8398 |
| **XGBoost** | **85.1%** | **0.9328** | **0.8496** | **0.8523** | **0.8509** |

XGBoost is selected as the best model with 85.1% accuracy and 0.9328 ROC-AUC.

### Confusion Matrix

A confusion matrix shows the 4 possible outcomes:

```
                    Predicted
                 No Churn    Churn
Actual  No Churn   [TN]      [FP]
        Churn      [FN]      [TP]
```

Confusion matrix PNGs are saved for each model at:
- `outputs/visualizations/confusion_matrix_logistic_regression.png`
- `outputs/visualizations/confusion_matrix_random_forest.png`
- `outputs/visualizations/confusion_matrix_xgboost.png`

### ROC Curve

The ROC curve plots True Positive Rate vs. False Positive Rate at every classification threshold. All three models are plotted on one chart for comparison.

Saved at: `outputs/visualizations/roc_curves_comparison.png`

### Feature Importance

The top features driving XGBoost predictions:

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|-----------------|
| 1 | `contract_type_Two year` | 0.152 | Two-year contracts dramatically reduce churn |
| 2 | `tenure_months` | 0.111 | Longer tenure = lower churn risk |
| 3 | `contract_type_One year` | 0.107 | One-year contracts moderately reduce churn |
| 4 | `support_intensity` | 0.107 | Frequent support tickets signal frustration |
| 5 | `monthly_charges` | 0.079 | Higher charges increase churn risk |
| 6 | `internet_service_Fiber optic` | 0.077 | Fiber optic customers churn more |
| 7 | `internet_service_No` | 0.073 | No internet = lower churn |
| 8 | `payment_method_Electronic check` | 0.056 | Electronic check users churn more |
| 9 | `charge_ratio` | 0.052 | Overpaying relative to peers increases risk |
| 10 | `online_security_Yes` | 0.035 | Security subscribers are stickier |

Feature importance chart saved at: `outputs/visualizations/feature_importance_xgboost.png`

---

## Model Persistence

### Saved Files

| File | Location | Description |
|------|----------|-------------|
| `logistic_regression.joblib` | `models/saved/` | Trained Logistic Regression model |
| `random_forest.joblib` | `models/saved/` | Trained Random Forest model |
| `xgboost.joblib` | `models/saved/` | Trained XGBoost model (best) |
| `scaler.joblib` | `models/saved/` | Fitted StandardScaler (needed for inference) |
| `*_metadata.json` | `models/saved/` | Training metadata (params, CV score, version, timestamp) |
| `*_v1.0.0_*.joblib` | `models/versioned/` | Timestamped model copies for audit trail |

### Loading a Model

```python
import joblib

# Load the best model
model = joblib.load("models/saved/xgboost.joblib")

# Load the scaler (required for inference)
scaler = joblib.load("models/saved/scaler.joblib")

# Load metadata
import json
with open("models/saved/xgboost_metadata.json") as f:
    meta = json.load(f)
print(meta)
# {
#   "model_name": "xgboost",
#   "best_cv_score": 0.9322,
#   "best_params": {"learning_rate": 0.1, "max_depth": 3, ...},
#   "version": "1.0.0",
#   "timestamp": "20260217_220843"
# }
```

### Model Versioning

Every training run saves versioned copies:
```
models/versioned/
├── xgboost_v1.0.0_20260217_220843.joblib
├── random_forest_v1.0.0_20260217_220831.joblib
└── logistic_regression_v1.0.0_20260217_220745.joblib
```

To rollback to a previous version:
```python
model = joblib.load("models/versioned/xgboost_v1.0.0_20260217_220843.joblib")
```

---

## Making Predictions

**File:** `src/models/predict.py`

### Single Prediction

```python
from src.models.predict import predict_churn

result = predict_churn({
    "age": 35,
    "tenure_months": 6,
    "monthly_charges": 85.0,
    "total_charges": 510.0,
    "num_support_tickets": 3,
    "monthly_minutes_used": 200,
    "data_usage_gb": 5.0,
})
# Returns: {"churn_probability": 0.83, "risk_level": "High"}
```

### Risk Level Classification

| Probability Range | Risk Level | Recommended Action |
|-------------------|------------|-------------------|
| 0.70 - 1.00 | **High** | Immediate outreach, retention offer |
| 0.40 - 0.69 | **Medium** | Proactive engagement, satisfaction survey |
| 0.00 - 0.39 | **Low** | Standard service, loyalty rewards |

### Batch Prediction

```python
from src.models.predict import batch_predict
import pandas as pd

customers = pd.read_csv("new_customers.csv")
results = batch_predict(customers)
# Returns DataFrame with churn_probability and risk_level columns
```

---

## Reports Generated

| Report | Location | Format | Contents |
|--------|----------|--------|----------|
| Model Comparison | `outputs/reports/model_comparison.csv` | CSV | Accuracy, ROC-AUC, F1 for all models |
| Model Comparison | `outputs/reports/model_comparison.json` | JSON | Full metrics including confusion matrices |
| Confusion Matrices | `outputs/visualizations/confusion_matrix_*.png` | PNG | Visual confusion matrix for each model |
| ROC Curves | `outputs/visualizations/roc_curves_comparison.png` | PNG | All models on one chart |
| Feature Importance | `outputs/visualizations/feature_importance_xgboost.png` | PNG | Top 15 features bar chart |
