"""
Prediction module for inference on new data.
"""
import joblib
import numpy as np
import pandas as pd

from src.config.settings import SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_production_model(model_name: str = "xgboost"):
    """Load the production model and scaler."""
    model = joblib.load(SAVED_MODELS_DIR / f"{model_name}.joblib")
    scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
    logger.info(f"Loaded production model: {model_name}")
    return model, scaler


def predict_churn(features: dict, model=None, scaler=None) -> dict:
    """
    Predict churn for a single customer.
    Returns churn probability and risk level.
    """
    if model is None or scaler is None:
        model, scaler = load_production_model()

    df = pd.DataFrame([features])

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    proba = model.predict_proba(df)[:, 1][0]

    if proba >= 0.7:
        risk_level = "High"
    elif proba >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    result = {
        "churn_probability": round(float(proba), 4),
        "risk_level": risk_level,
    }

    logger.info(f"Prediction: {result}")
    return result


def batch_predict(df: pd.DataFrame, model=None, scaler=None) -> pd.DataFrame:
    """Predict churn for a batch of customers."""
    if model is None or scaler is None:
        model, scaler = load_production_model()

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    probas = model.predict_proba(df)[:, 1]
    results = pd.DataFrame({
        "churn_probability": np.round(probas, 4),
        "risk_level": pd.cut(probas, bins=[-1, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"]),
    })
    return results
