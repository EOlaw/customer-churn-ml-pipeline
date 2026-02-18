"""
Tests for the customer churn ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import generate_synthetic_data
from src.data.feature_engineering import engineer_features
from src.data.preprocessing import handle_missing_values, encode_categorical
from src.utils.validation import validate_raw_data


class TestDataIngestion:
    def test_generates_correct_number_of_records(self):
        df = generate_synthetic_data(n_samples=1000, seed=42)
        assert len(df) == 1000

    def test_has_required_columns(self):
        df = generate_synthetic_data(n_samples=100, seed=42)
        required = ["customer_id", "gender", "age", "tenure_months",
                     "monthly_charges", "churn"]
        for col in required:
            assert col in df.columns

    def test_churn_is_binary(self):
        df = generate_synthetic_data(n_samples=1000, seed=42)
        assert set(df["churn"].dropna().unique()).issubset({0, 1})

    def test_has_missing_values(self):
        df = generate_synthetic_data(n_samples=10000, seed=42)
        assert df.isnull().any().any(), "Should have injected missing values"

    def test_customer_ids_unique(self):
        df = generate_synthetic_data(n_samples=1000, seed=42)
        assert df["customer_id"].nunique() == 1000


class TestFeatureEngineering:
    def test_creates_new_features(self):
        df = generate_synthetic_data(n_samples=100, seed=42)
        result = engineer_features(df)
        new_cols = ["tenure_bucket", "avg_monthly_spend", "engagement_score",
                    "charge_ratio", "support_intensity"]
        for col in new_cols:
            assert col in result.columns

    def test_engagement_score_bounded(self):
        df = generate_synthetic_data(n_samples=1000, seed=42)
        result = engineer_features(df)
        assert result["engagement_score"].min() >= 0
        assert result["engagement_score"].max() <= 1


class TestPreprocessing:
    def test_handle_missing_values(self):
        df = generate_synthetic_data(n_samples=1000, seed=42)
        result = handle_missing_values(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isnull().any().any()

    def test_encode_categorical(self):
        df = generate_synthetic_data(n_samples=100, seed=42)
        df = handle_missing_values(df)
        result, mapping = encode_categorical(df)
        obj_cols = result.select_dtypes(include=["object"]).columns
        # Only customer_id should remain as object
        assert list(obj_cols) == ["customer_id"]


class TestValidation:
    def test_validates_good_data(self):
        df = generate_synthetic_data(n_samples=100, seed=42)
        assert validate_raw_data(df) is True

    def test_fails_on_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert validate_raw_data(df) is False
