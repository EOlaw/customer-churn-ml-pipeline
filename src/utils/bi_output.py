"""
Business Intelligence output generation.
Creates Tableau-ready datasets, segment summaries, and executive KPI reports.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.settings import TABLEAU_DIR, REPORTS_DIR, VISUALIZATIONS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_tableau_dataset(df: pd.DataFrame, labels: np.ndarray = None) -> None:
    """Export a Tableau-ready CSV with all features and segments."""
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    export = df.copy()

    if labels is not None:
        # Align labels with dataframe (clustering may drop some rows)
        if len(labels) == len(export):
            export["customer_segment"] = labels
        else:
            export["customer_segment"] = np.nan
            export.iloc[:len(labels), export.columns.get_loc("customer_segment")] = labels

    path = TABLEAU_DIR / "churn_analysis_tableau.csv"
    export.to_csv(path, index=False)
    logger.info(f"Tableau dataset exported to {path}")


def generate_segment_summary(segment_profile: pd.DataFrame) -> None:
    """Export formatted segment summary for BI tools."""
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLEAU_DIR / "segment_summary.csv"
    segment_profile.to_csv(path)
    logger.info(f"Segment summary exported to {path}")


def generate_executive_kpi_summary(
    raw_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    segment_profile: pd.DataFrame,
) -> dict:
    """Generate executive KPI summary report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total_customers = len(raw_df)
    churn_rate = raw_df["churn"].mean() if "churn" in raw_df.columns else None
    avg_revenue = raw_df["monthly_charges"].mean() if "monthly_charges" in raw_df.columns else None
    total_revenue = raw_df["total_charges"].sum() if "total_charges" in raw_df.columns else None

    best_model = comparison_df.loc[comparison_df["roc_auc"].idxmax()]

    kpi = {
        "business_metrics": {
            "total_customers": int(total_customers),
            "overall_churn_rate": round(float(churn_rate), 4) if churn_rate else None,
            "avg_monthly_revenue_per_customer": round(float(avg_revenue), 2) if avg_revenue else None,
            "total_revenue": round(float(total_revenue), 2) if total_revenue else None,
            "at_risk_customers": int((raw_df["churn"] == 1).sum()) if "churn" in raw_df.columns else None,
            "potential_revenue_at_risk": round(
                float(raw_df[raw_df["churn"] == 1]["monthly_charges"].sum() * 12), 2
            ) if "churn" in raw_df.columns else None,
        },
        "model_performance": {
            "best_model": best_model["model_name"],
            "accuracy": float(best_model["accuracy"]),
            "roc_auc": float(best_model["roc_auc"]),
            "precision": float(best_model["precision"]),
            "recall": float(best_model["recall"]),
            "f1_score": float(best_model["f1_score"]),
        },
        "segmentation": {
            "num_segments": len(segment_profile),
            "segments": segment_profile[["business_label", "count", "churn_rate"]].to_dict("records")
            if "business_label" in segment_profile.columns else [],
        },
    }

    with open(REPORTS_DIR / "executive_kpi_summary.json", "w") as f:
        json.dump(kpi, f, indent=2, default=str)

    # Also export as CSV for Tableau
    kpi_flat = pd.DataFrame([{
        "Total Customers": kpi["business_metrics"]["total_customers"],
        "Churn Rate": kpi["business_metrics"]["overall_churn_rate"],
        "Avg Monthly Revenue": kpi["business_metrics"]["avg_monthly_revenue_per_customer"],
        "Total Revenue": kpi["business_metrics"]["total_revenue"],
        "At-Risk Customers": kpi["business_metrics"]["at_risk_customers"],
        "Revenue At Risk (Annual)": kpi["business_metrics"]["potential_revenue_at_risk"],
        "Best Model": kpi["model_performance"]["best_model"],
        "Model ROC-AUC": kpi["model_performance"]["roc_auc"],
        "Model Accuracy": kpi["model_performance"]["accuracy"],
    }])
    kpi_flat.to_csv(TABLEAU_DIR / "executive_kpi.csv", index=False)

    logger.info("Executive KPI summary generated")
    return kpi


def generate_churn_distribution_chart(df: pd.DataFrame) -> None:
    """Create churn distribution visualization."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if "churn" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Churn distribution
    df["churn"].value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Churn Distribution")
    axes[0].set_xticklabels(["No Churn", "Churn"], rotation=0)
    axes[0].set_ylabel("Count")

    # Monthly charges by churn
    df.boxplot(column="monthly_charges", by="churn", ax=axes[1])
    axes[1].set_title("Monthly Charges by Churn Status")
    axes[1].set_xlabel("Churn")
    plt.suptitle("")

    # Tenure by churn
    df.boxplot(column="tenure_months", by="churn", ax=axes[2])
    axes[2].set_title("Tenure by Churn Status")
    axes[2].set_xlabel("Churn")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "churn_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved churn distribution chart")


def generate_all_bi_outputs(
    raw_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    segment_profile: pd.DataFrame,
    labels: np.ndarray = None,
) -> dict:
    """Generate all BI outputs."""
    logger.info("Generating BI outputs...")
    generate_tableau_dataset(raw_df, labels)
    generate_segment_summary(segment_profile)
    kpi = generate_executive_kpi_summary(raw_df, comparison_df, segment_profile)
    generate_churn_distribution_chart(raw_df)
    logger.info("All BI outputs generated")
    return kpi
