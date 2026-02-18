"""
End-to-end training pipeline orchestrator.
Runs the full ML pipeline: ingest → engineer → preprocess → train → evaluate → cluster.
"""
import joblib
import time
from src.config.settings import SAVED_MODELS_DIR
from src.data.ingestion import ingest_data
from src.data.feature_engineering import engineer_features
from src.data.preprocessing import preprocess
from src.models.train import split_data, train_all_models, get_feature_importance
from src.models.evaluate import evaluate_all_models, select_best_model, plot_feature_importance
from src.models.clustering import run_clustering
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline() -> dict:
    """Execute the complete training pipeline."""
    start = time.time()
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Data Ingestion
    logger.info("\n[1/6] Data Ingestion")
    raw_df = ingest_data()

    # Step 2: Feature Engineering (on raw data before encoding)
    logger.info("\n[2/6] Feature Engineering")
    engineered_df = engineer_features(raw_df)

    # Step 3: Preprocessing
    logger.info("\n[3/6] Preprocessing")
    processed_df, scaler, encoding_map = preprocess(engineered_df)

    # Save scaler for inference
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.joblib")

    # Step 4: Train Models
    logger.info("\n[4/6] Model Training")
    X_train, X_test, y_train, y_test = split_data(processed_df)
    models_dict = train_all_models(X_train, y_train)

    # Step 5: Evaluate Models
    logger.info("\n[5/6] Model Evaluation")
    comparison = evaluate_all_models(models_dict, X_test, y_test)
    best_name, best_model = select_best_model(comparison, models_dict)

    # Feature importance for best model
    fi = get_feature_importance(best_model, X_train.columns.tolist(), best_name)
    if not fi.empty:
        plot_feature_importance(fi, best_name)

    # Step 6: Clustering
    logger.info("\n[6/6] Customer Segmentation")
    labels, segment_profile = run_clustering(raw_df, n_clusters=4)

    elapsed = round(time.time() - start, 1)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PIPELINE COMPLETE in {elapsed}s")
    logger.info(f"Best model: {best_name}")
    logger.info(f"{'=' * 60}")

    return {
        "best_model_name": best_name,
        "comparison": comparison,
        "segment_profile": segment_profile,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    run_pipeline()
