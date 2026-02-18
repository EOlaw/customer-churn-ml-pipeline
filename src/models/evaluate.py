"""
Model evaluation module.
Generates classification metrics, confusion matrices, and ROC curves.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

from src.config.settings import REPORTS_DIR, VISUALIZATIONS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute full evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {metrics['accuracy']}")
    logger.info(f"Precision: {metrics['precision']}")
    logger.info(f"Recall:    {metrics['recall']}")
    logger.info(f"F1 Score:  {metrics['f1_score']}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """Save confusion matrix as a PNG."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.colorbar(im)

    classes = ["No Churn", "Churn"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix for {model_name}")


def plot_roc_curves(models_dict: dict, X_test, y_test) -> None:
    """Plot ROC curves for all models on one figure."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (model, _) in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "roc_curves_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved ROC curves comparison")


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str, top_n: int = 15) -> None:
    """Save feature importance bar chart."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top["importance"].values, align="center")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features - {model_name}")
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / f"feature_importance_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"Saved feature importance chart for {model_name}")


def evaluate_all_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and generate comparison report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for name, (model, _) in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)

    plot_roc_curves(models_dict, X_test, y_test)

    # Save comparison report
    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info(f"\nModel Comparison:\n{comparison[['model_name', 'accuracy', 'roc_auc', 'f1_score']].to_string()}")
    return comparison


def select_best_model(comparison_df: pd.DataFrame, models_dict: dict) -> tuple:
    """Select the best model based on ROC-AUC score."""
    best_row = comparison_df.loc[comparison_df["roc_auc"].idxmax()]
    best_name = best_row["model_name"]
    best_model = models_dict[best_name][0]
    logger.info(f"Best model: {best_name} (ROC-AUC: {best_row['roc_auc']})")
    return best_name, best_model
