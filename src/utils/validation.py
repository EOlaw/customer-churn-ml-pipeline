"""
Data validation utilities.
"""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "customer_id", "gender", "age", "tenure_months", "contract_type",
    "monthly_charges", "total_charges", "payment_method",
    "num_support_tickets", "internet_service", "online_security",
    "tech_support", "streaming_tv", "streaming_movies",
    "paperless_billing", "num_dependents", "partner",
    "monthly_minutes_used", "data_usage_gb", "churn",
]


def validate_raw_data(df: pd.DataFrame) -> bool:
    """Validate that raw data has expected schema and quality."""
    errors = []

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("DataFrame is empty")

    if "customer_id" in df.columns and df.duplicated(subset=["customer_id"]).any():
        n_dupes = df.duplicated(subset=["customer_id"]).sum()
        errors.append(f"Found {n_dupes} duplicate customer_ids")

    numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges"]
    for col in numeric_cols:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative values in {col}")

    if errors:
        for e in errors:
            logger.error(f"Validation error: {e}")
        return False

    logger.info("Raw data validation passed")
    return True


def validate_processed_data(df: pd.DataFrame) -> bool:
    """Validate processed data is model-ready."""
    errors = []

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Null values remain in: {null_cols}")

    if errors:
        for e in errors:
            logger.error(f"Validation error: {e}")
        return False

    logger.info("Processed data validation passed")
    return True
