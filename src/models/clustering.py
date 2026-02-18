"""
Customer behavioral segmentation using K-Means clustering.
Includes elbow method, silhouette analysis, and segment interpretation.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config.settings import VISUALIZATIONS_DIR, REPORTS_DIR, SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLUSTERING_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets", "monthly_minutes_used", "data_usage_gb",
]


def prepare_clustering_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare and scale data for clustering."""
    available = [c for c in CLUSTERING_FEATURES if c in df.columns]
    cluster_df = df[available].copy().dropna()

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(cluster_df),
        columns=available,
        index=cluster_df.index,
    )
    return scaled_data, scaler


def elbow_method(data: pd.DataFrame, max_k: int = 10) -> list[float]:
    """Run elbow method to find optimal K."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(data)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "bo-")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal K")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "elbow_method.png", dpi=150)
    plt.close()
    logger.info("Saved elbow method plot")

    return inertias


def silhouette_analysis(data: pd.DataFrame, max_k: int = 10) -> dict[int, float]:
    """Compute silhouette scores for different K values."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    scores = {}
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        scores[k] = round(score, 4)
        logger.info(f"K={k}: Silhouette Score = {score:.4f}")

    best_k = max(scores, key=scores.get)
    logger.info(f"Best K by silhouette score: {best_k} (score={scores[best_k]})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(scores.keys()), list(scores.values()), "ro-")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    ax.axvline(x=best_k, color="green", linestyle="--", label=f"Best K={best_k}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "silhouette_analysis.png", dpi=150)
    plt.close()

    return scores


def fit_kmeans(data: pd.DataFrame, n_clusters: int = 4) -> tuple[KMeans, np.ndarray]:
    """Fit K-Means with the chosen K."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(data)
    score = silhouette_score(data, labels)
    logger.info(f"K-Means fit with K={n_clusters}, Silhouette={score:.4f}")
    return km, labels


def interpret_segments(df_original: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Generate business-readable segment interpretation."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = df_original.copy()
    available = [c for c in CLUSTERING_FEATURES if c in df.columns]
    df = df.loc[df[available].dropna().index].copy()
    df["segment"] = labels

    # Compute segment profiles
    profile = df.groupby("segment").agg(
        count=("segment", "size"),
        avg_tenure=("tenure_months", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_total_charges=("total_charges", "mean"),
        avg_support_tickets=("num_support_tickets", "mean"),
        avg_minutes=("monthly_minutes_used", "mean"),
        avg_data_gb=("data_usage_gb", "mean"),
    ).round(2)

    if "churn" in df.columns:
        churn_rate = df.groupby("segment")["churn"].mean().round(3)
        profile["churn_rate"] = churn_rate

    # Business labels — rank segments by churn risk and characteristics
    segment_labels = {}
    ranked = profile.sort_values("churn_rate", ascending=True) if "churn_rate" in profile.columns else profile
    for i, seg in enumerate(ranked.index):
        row = profile.loc[seg]
        churn = row.get("churn_rate", 0.5)
        if row["avg_tenure"] > 36 and churn < profile["churn_rate"].median():
            segment_labels[seg] = "Loyal High-Value"
        elif churn > profile["churn_rate"].quantile(0.75):
            if row["avg_monthly_charges"] > profile["avg_monthly_charges"].median():
                segment_labels[seg] = "At-Risk / High Spenders"
            else:
                segment_labels[seg] = "At-Risk / Frustrated"
        elif row["avg_monthly_charges"] < profile["avg_monthly_charges"].quantile(0.3):
            segment_labels[seg] = "Budget-Conscious"
        elif row["avg_data_gb"] > profile["avg_data_gb"].median() * 1.5:
            segment_labels[seg] = "New Power Users"
        else:
            segment_labels[seg] = "Mid-Tier Stable"

    profile["business_label"] = profile.index.map(segment_labels)

    profile.to_csv(REPORTS_DIR / "segment_profiles.csv")
    logger.info(f"\nSegment Profiles:\n{profile.to_string()}")

    return profile


def run_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple[np.ndarray, pd.DataFrame]:
    """Full clustering pipeline."""
    logger.info("Starting clustering pipeline...")

    scaled_data, scaler = prepare_clustering_data(df)
    elbow_method(scaled_data)
    silhouette_analysis(scaled_data)

    km, labels = fit_kmeans(scaled_data, n_clusters)
    profile = interpret_segments(df, labels)

    # Save model
    import joblib
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, SAVED_MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, SAVED_MODELS_DIR / "clustering_scaler.joblib")

    return labels, profile
