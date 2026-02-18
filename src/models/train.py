"""
Model training module.
Trains Logistic Regression, Random Forest, and XGBoost classifiers
with cross-validation and hyperparameter tuning.
"""
import joblib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from src.config.settings import (
    SAVED_MODELS_DIR, VERSIONED_MODELS_DIR,
    TEST_SIZE, RANDOM_SEED, CV_FOLDS, SCORING_METRIC, MODEL_VERSION,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_CONFIGS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "param_grid": {
            "C": [0.01, 0.1, 1.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_SEED, eval_metric="logloss",
            n_jobs=-1,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
}


def split_data(df: pd.DataFrame, target_col: str = "churn"):
    """Split data into train/test sets, excluding customer_id."""
    feature_cols = [c for c in df.columns if c not in [target_col, "customer_id"]]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def train_model(model_name: str, X_train, y_train) -> tuple:
    """
    Train a single model with GridSearchCV hyperparameter tuning.
    Returns: (best_model, cv_results)
    """
    config = MODEL_CONFIGS[model_name]
    logger.info(f"Training {model_name} with {CV_FOLDS}-fold CV...")

    grid_search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    logger.info(f"{model_name} | Best CV {SCORING_METRIC}: {best_score:.4f}")
    logger.info(f"{model_name} | Best params: {best_params}")

    return best_model, {
        "model_name": model_name,
        "best_cv_score": best_score,
        "best_params": best_params,
    }


def save_model(model, model_name: str, metadata: dict) -> Path:
    """Save trained model and metadata with versioning."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to saved directory (latest)
    model_path = SAVED_MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    # Save versioned copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = VERSIONED_MODELS_DIR / f"{model_name}_v{MODEL_VERSION}_{timestamp}.joblib"
    joblib.dump(model, versioned_path)

    # Save metadata
    metadata["version"] = MODEL_VERSION
    metadata["timestamp"] = timestamp
    metadata["model_path"] = str(model_path)
    meta_path = SAVED_MODELS_DIR / f"{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved {model_name} to {model_path}")
    return model_path


def load_model(model_name: str):
    """Load a saved model."""
    model_path = SAVED_MODELS_DIR / f"{model_name}.joblib"
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top 10 features for {model_name}:")
    for _, row in fi.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return fi


def train_all_models(X_train, y_train) -> dict:
    """Train all models and return dict of {name: (model, cv_results)}."""
    results = {}
    for name in MODEL_CONFIGS:
        model, cv_results = train_model(name, X_train, y_train)
        save_model(model, name, cv_results.copy())
        results[name] = (model, cv_results)
    return results
