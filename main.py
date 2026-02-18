"""
Main entry point for the Customer Churn ML Pipeline.
Run this to execute the full pipeline and generate all outputs.
"""
import sys
from src.pipelines.training_pipeline import run_pipeline
from src.utils.bi_output import generate_all_bi_outputs
from src.data.ingestion import load_from_csv
from src.config.settings import RAW_DATA_DIR, REPORTS_DIR
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def main():
    logger.info("Starting Customer Churn ML Pipeline...")

    # Run the full training pipeline
    results = run_pipeline()

    # Generate BI outputs
    raw_df = load_from_csv(RAW_DATA_DIR / "customer_churn_raw.csv")
    comparison_df = results["comparison"]
    segment_profile = results["segment_profile"]

    kpi = generate_all_bi_outputs(
        raw_df=raw_df,
        comparison_df=comparison_df,
        segment_profile=segment_profile,
    )

    # Print executive summary
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    bm = kpi["business_metrics"]
    mp = kpi["model_performance"]
    print(f"Total Customers:        {bm['total_customers']:,}")
    print(f"Overall Churn Rate:     {bm['overall_churn_rate']:.1%}")
    print(f"At-Risk Customers:      {bm['at_risk_customers']:,}")
    print(f"Revenue At Risk (Annual): ${bm['potential_revenue_at_risk']:,.2f}")
    print(f"\nBest Model:             {mp['best_model']}")
    print(f"Accuracy:               {mp['accuracy']:.1%}")
    print(f"ROC-AUC:                {mp['roc_auc']:.4f}")
    print(f"Precision:              {mp['precision']:.4f}")
    print(f"Recall:                 {mp['recall']:.4f}")
    print(f"\nPipeline completed in {results['elapsed_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
