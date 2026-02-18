"""
Data ingestion module.
Generates synthetic customer churn data (100K+ records) and handles CSV/S3 simulation.
"""
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from src.config.settings import (
    RAW_DATA_DIR, S3_SIMULATION_DIR, N_SAMPLES, RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic customer churn dataset.

    Creates 100K+ records with correlated features that mimic real-world
    telecom/subscription churn patterns.
    """
    np.random.seed(seed)
    logger.info(f"Generating {n_samples} synthetic customer records...")

    customer_ids = [f"CUST-{i:06d}" for i in range(1, n_samples + 1)]

    # Demographics
    gender = np.random.choice(["Male", "Female"], n_samples)
    age = np.clip(np.random.normal(45, 15, n_samples).astype(int), 18, 85)
    partner = np.random.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    num_dependents = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    # Account info
    tenure_months = np.clip(np.random.exponential(30, n_samples).astype(int), 1, 72)
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.50, 0.25, 0.25],
    )

    # Billing
    monthly_charges = np.round(np.random.uniform(18.0, 120.0, n_samples), 2)
    # Higher tenure => higher total charges (correlated)
    total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.85, 1.15, n_samples), 2)

    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples,
        p=[0.35, 0.20, 0.22, 0.23],
    )
    paperless_billing = np.random.choice(["Yes", "No"], n_samples, p=[0.60, 0.40])

    # Services
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    online_security = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65]),
    )
    tech_support = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70]),
    )
    streaming_tv = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.40, 0.60]),
    )
    streaming_movies = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.40, 0.60]),
    )

    # Usage metrics
    num_support_tickets = np.random.poisson(1.5, n_samples)
    monthly_minutes_used = np.clip(np.random.normal(450, 150, n_samples).astype(int), 0, 1500)
    data_usage_gb = np.round(np.random.exponential(8, n_samples), 2)

    # ---- Churn label (correlated with features) ----
    churn_score = np.zeros(n_samples)

    # Month-to-month contracts churn more (strong signal)
    churn_score += np.where(np.array(contract_type) == "Month-to-month", 3.0, 0)
    churn_score += np.where(np.array(contract_type) == "One year", 0.5, 0)
    churn_score -= np.where(np.array(contract_type) == "Two year", 2.0, 0)

    # Short tenure => higher churn (strong signal)
    churn_score += np.where(tenure_months < 6, 2.5, 0)
    churn_score += np.where(tenure_months < 12, 1.0, 0)
    churn_score -= np.where(tenure_months > 36, 1.5, 0)
    churn_score -= np.where(tenure_months > 48, 1.0, 0)

    # High monthly charges (strong signal)
    churn_score += np.where(monthly_charges > 80, 1.5, 0)
    churn_score += np.where(monthly_charges > 100, 1.0, 0)
    churn_score -= np.where(monthly_charges < 40, 1.0, 0)

    # Electronic check payment
    churn_score += np.where(np.array(payment_method) == "Electronic check", 1.2, 0)

    # Fiber optic (often has issues in real data)
    churn_score += np.where(np.array(internet_service) == "Fiber optic", 1.0, 0)
    churn_score -= np.where(np.array(internet_service) == "No", 0.5, 0)

    # No online security or tech support
    churn_score += np.where(np.array(online_security) == "No", 0.8, 0)
    churn_score += np.where(np.array(tech_support) == "No", 0.6, 0)

    # Support tickets (strong signal)
    churn_score += num_support_tickets * 0.6

    # Low usage
    churn_score += np.where(monthly_minutes_used < 200, 1.0, 0)
    churn_score -= np.where(monthly_minutes_used > 600, 0.5, 0)

    # No partner/dependents
    churn_score += np.where(np.array(partner) == "No", 0.5, 0)
    churn_score += np.where(num_dependents == 0, 0.5, 0)

    # Add noise (minimal for learnable signal)
    churn_score += np.random.normal(0, 0.3, n_samples)

    # Convert to probability via sigmoid (steep curve for clear separation)
    churn_prob = 1 / (1 + np.exp(-1.0 * (churn_score - np.median(churn_score))))
    churn = (np.random.uniform(0, 1, n_samples) < churn_prob).astype(int)

    # Inject ~3% missing values in select columns
    for col_vals in [monthly_charges, total_charges, data_usage_gb]:
        mask = np.random.random(n_samples) < 0.03
        col_vals[mask] = np.nan

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": gender,
        "age": age,
        "partner": partner,
        "num_dependents": num_dependents,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "num_support_tickets": num_support_tickets,
        "monthly_minutes_used": monthly_minutes_used,
        "data_usage_gb": data_usage_gb,
        "churn": churn,
    })

    churn_rate = df["churn"].mean()
    logger.info(f"Generated {len(df)} records | Churn rate: {churn_rate:.1%}")
    return df


def save_to_csv(df: pd.DataFrame, filename: str = "customer_churn_raw.csv") -> Path:
    """Save DataFrame to CSV in the raw data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved raw data to {filepath}")
    return filepath


def simulate_s3_upload(source_path: Path, s3_prefix: str = "raw") -> Path:
    """Simulate uploading a file to S3 by copying to the S3 simulation directory."""
    dest_dir = S3_SIMULATION_DIR / s3_prefix
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    logger.info(f"Simulated S3 upload: s3://{dest_dir.name}/{s3_prefix}/{source_path.name}")
    return dest_path


def load_from_csv(filepath: Path) -> pd.DataFrame:
    """Load data from CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def ingest_data() -> pd.DataFrame:
    """Full ingestion pipeline: generate, save, simulate S3 upload."""
    df = generate_synthetic_data()
    csv_path = save_to_csv(df)
    simulate_s3_upload(csv_path)
    return df


if __name__ == "__main__":
    ingest_data()
