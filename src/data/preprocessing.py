"""
Data preprocessing module.
Handles cleaning, encoding, scaling, and SQL storage.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

from src.config.settings import DATABASE_PATH, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.validation import validate_raw_data, validate_processed_data

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using median for numeric columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One-hot encode categorical features."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Exclude customer_id from encoding
    if "customer_id" in categorical_cols:
        categorical_cols.remove("customer_id")

    encoding_map = {}
    for col in categorical_cols:
        encoding_map[col] = df[col].unique().tolist()

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns → {len(df_encoded.columns)} total columns")

    return df_encoded, encoding_map


def scale_features(df: pd.DataFrame, target_col: str = "churn") -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features using StandardScaler."""
    df = df.copy()
    exclude_cols = [target_col, "customer_id"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"Scaled {len(numeric_cols)} numeric features")

    return df, scaler


def store_to_sql(df: pd.DataFrame, table_name: str = "customer_data") -> None:
    """Store processed data into SQLite database."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Stored {len(df)} records to SQL table '{table_name}'")


def export_processed_csv(df: pd.DataFrame, filename: str = "customer_churn_processed.csv") -> Path:
    """Export processed data as CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Exported processed data to {filepath}")
    return filepath


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, dict]:
    """
    Full preprocessing pipeline.
    Returns: (processed_df, scaler, encoding_map)
    """
    logger.info("Starting preprocessing pipeline...")
    validate_raw_data(df)

    df = handle_missing_values(df)
    df, encoding_map = encode_categorical(df)
    df, scaler = scale_features(df)

    validate_processed_data(df)

    # Store to SQL and export CSV
    store_to_sql(df)
    export_processed_csv(df)

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df, scaler, encoding_map
