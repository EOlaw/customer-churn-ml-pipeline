"""
Feature engineering module.
Creates derived business features from raw customer data.
"""
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Create categorical tenure buckets."""
    df = df.copy()
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr"]
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, include_lowest=True)
    logger.info("Created tenure_bucket feature")
    return df


def create_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average monthly spend over tenure."""
    df = df.copy()
    df["avg_monthly_spend"] = np.where(
        df["tenure_months"] > 0,
        np.round(df["total_charges"] / df["tenure_months"], 2),
        df["monthly_charges"],
    )
    logger.info("Created avg_monthly_spend feature")
    return df


def create_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite engagement score based on usage and service adoption.
    Higher score = more engaged customer.
    """
    df = df.copy()

    # Normalize usage metrics to 0-1 range
    minutes_norm = df["monthly_minutes_used"] / df["monthly_minutes_used"].max()
    data_norm = df["data_usage_gb"] / df["data_usage_gb"].max()

    # Count service subscriptions
    service_cols = ["online_security", "tech_support", "streaming_tv", "streaming_movies"]
    service_count = sum(
        (df[col] == "Yes").astype(int) if col in df.columns else 0
        for col in service_cols
    )
    service_norm = service_count / len(service_cols)

    # Weighted composite
    df["engagement_score"] = np.round(
        0.3 * minutes_norm + 0.3 * data_norm + 0.4 * service_norm, 4
    )
    logger.info("Created engagement_score feature")
    return df


def create_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of monthly charges to average for the customer's contract type."""
    df = df.copy()
    avg_by_contract = df.groupby("contract_type")["monthly_charges"].transform("mean")
    df["charge_ratio"] = np.round(df["monthly_charges"] / avg_by_contract, 4)
    logger.info("Created charge_ratio feature")
    return df


def create_support_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Support tickets per month of tenure."""
    df = df.copy()
    df["support_intensity"] = np.where(
        df["tenure_months"] > 0,
        np.round(df["num_support_tickets"] / df["tenure_months"], 4),
        df["num_support_tickets"].astype(float),
    )
    logger.info("Created support_intensity feature")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    logger.info("Starting feature engineering...")

    df = create_tenure_buckets(df)
    df = create_avg_monthly_spend(df)
    df = create_engagement_score(df)
    df = create_charge_ratio(df)
    df = create_support_intensity(df)

    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df
