"""
FastAPI inference endpoint for churn prediction.
"""
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import SAVED_MODELS_DIR, MODEL_VERSION
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability and risk level",
    version=MODEL_VERSION,
)

# Global model references (loaded at startup)
model = None
scaler = None
feature_names = None


class CustomerFeatures(BaseModel):
    """Input schema for churn prediction."""
    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_tickets: int = Field(..., ge=0)
    monthly_minutes_used: int = Field(..., ge=0)
    data_usage_gb: float = Field(..., ge=0)
    num_dependents: int = Field(..., ge=0)
    contract_type_One_year: int = Field(0, ge=0, le=1)
    contract_type_Two_year: int = Field(0, ge=0, le=1)
    internet_service_Fiber_optic: int = Field(0, ge=0, le=1)
    internet_service_No: int = Field(0, ge=0, le=1)
    paperless_billing_Yes: int = Field(0, ge=0, le=1)

    model_config = {"json_schema_extra": {
        "examples": [{
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
            "paperless_billing_Yes": 1,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema for churn prediction."""
    churn_probability: float
    risk_level: str
    model_version: str


@app.on_event("startup")
def load_models():
    """Load model and scaler at startup."""
    global model, scaler, feature_names
    try:
        model = joblib.load(SAVED_MODELS_DIR / "xgboost.joblib")
        scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
        # Get expected feature names from scaler
        feature_names = scaler.feature_names_in_.tolist() if hasattr(scaler, "feature_names_in_") else None
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("Models not found. Run the training pipeline first.")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict-churn", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    """Predict churn probability for a customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    features = customer.model_dump()
    df = pd.DataFrame([features])

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Align features with what the model expects
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    try:
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        proba = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if proba >= 0.7:
        risk_level = "High"
    elif proba >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_level=risk_level,
        model_version=MODEL_VERSION,
    )


@app.get("/model-info")
def model_info():
    """Return model metadata."""
    import json
    meta_path = SAVED_MODELS_DIR / "xgboost_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"message": "No model metadata available"}
