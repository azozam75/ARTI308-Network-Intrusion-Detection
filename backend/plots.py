"""Week-8 evaluation plots for the tuned IForest bundle.

Consumes the scored parquets produced by training + test evaluation and
the saved threshold from ``run_summary.json``. Produces:

  outputs/figures/roc_curve.png          — val + test ROC overlay, with
                                           the operating point marked
  outputs/figures/confusion_matrix.png   — 2×2 heatmap on the test set
  outputs/figures/score_distributions.png — per-category score boxplots
                                            on the test set, with the
                                            flagging threshold drawn

Run after ``backend.model`` (produces ``val_scored.parquet``) and
``backend.evaluation`` (produces ``test_scored.parquet`` + merges test
metrics into ``run_summary.json``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
BENIGN_LABEL = "BENIGN"


def _binary_labels(categories: np.ndarray) -> np.ndarray:
    return (categories != BENIGN_LABEL).astype(np.int8)


def plot_roc_curves(
    val: pd.DataFrame, test: pd.DataFrame, threshold: float, out_path: Path
) -> dict[str, float]:
    """Overlay val + test ROC; mark the chosen-threshold operating point on test."""
    y_val = _binary_labels(val["attack_category"].to_numpy())
    y_test = _binary_labels(test["attack_category"].to_numpy())
    val_fpr, val_tpr, _ = roc_curve(y_val, val["score"].to_numpy())
    test_fpr, test_tpr, _ = roc_curve(y_test, test["score"].to_numpy())
    val_auc, test_auc = auc(val_fpr, val_tpr), auc(test_fpr, test_tpr)

    test_preds = (test["score"].to_numpy() >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    op_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    op_tpr = tp / (tp + fn) if (tp + fn) else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(val_fpr, val_tpr, color="steelblue", lw=1.5, label=f"Val  (AUC = {val_auc:.3f})")
    ax.plot(test_fpr, test_tpr, color="darkorange", lw=1.5, label=f"Test (AUC = {test_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.6, label="Chance")
    ax.scatter(
        [op_fpr], [op_tpr], s=90, color="darkorange", edgecolor="black",
        zorder=5, label=f"Operating point (thr = {threshold:.4f})",
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — tuned IForest on CIC-IDS-2017")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {"val_auc": float(val_auc), "test_auc": float(test_auc)}


def plot_confusion_matrix(
    test: pd.DataFrame, threshold: float, out_path: Path
) -> None:
    y_true = _binary_labels(test["attack_category"].to_numpy())
    y_pred = (test["score"].to_numpy() >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Normalized view — proportions per actual row — reads easier given the
    # ~4× BENIGN/attack imbalance. Raw counts go into the annotations.
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    annot = np.array(
        [[f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)" for j in range(2)] for i in range(2)]
    )
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
        xticklabels=["Predicted BENIGN", "Predicted Attack"],
        yticklabels=["Actual BENIGN", "Actual Attack"],
        cbar_kws={"label": "Row-normalized rate"}, ax=ax,
    )
    ax.set_title(f"Confusion matrix (test set, threshold = {threshold:.4f})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_score_distributions(
    test: pd.DataFrame, threshold: float, out_path: Path
) -> None:
    """Per-category score boxplots with the threshold drawn as a reference line.

    Categories are ordered by median score (lowest → highest), so the
    visual left-to-right sweep goes from 'cleanly BENIGN-like' to 'most
    anomalous'. The gap from the threshold line to each box directly
    shows why stealthy attacks are missed under a global threshold.
    """
    order = (
        test.groupby("attack_category", observed=True)["score"]
        .median()
        .sort_values()
        .index.tolist()
    )
    n_cats = len(order)

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * n_cats + 3), 6))
    palette = sns.color_palette("tab10", n_colors=n_cats)
    sns.boxplot(
        data=test, x="attack_category", y="score", order=order,
        hue="attack_category", hue_order=order, palette=palette,
        legend=False, showfliers=False, ax=ax,
    )
    ax.axhline(
        threshold, color="red", linestyle="--", lw=1.2,
        label=f"Flag threshold = {threshold:.4f}",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Anomaly score  (higher = more anomalous)")
    ax.set_title("Score distribution by attack category (test set)")
    ax.legend(loc="upper left", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])

    LOGGER.info("Loading scored splits")
    val = pd.read_parquet(RESULTS_DIR / "val_scored.parquet")
    test = pd.read_parquet(RESULTS_DIR / "test_scored.parquet")

    LOGGER.info("ROC curve")
    auc_stats = plot_roc_curves(val, test, threshold, FIGURES_DIR / "roc_curve.png")
    LOGGER.info("  val AUC=%.4f  test AUC=%.4f", auc_stats["val_auc"], auc_stats["test_auc"])

    LOGGER.info("Confusion matrix")
    plot_confusion_matrix(test, threshold, FIGURES_DIR / "confusion_matrix.png")

    LOGGER.info("Score distributions")
    plot_score_distributions(test, threshold, FIGURES_DIR / "score_distributions.png")

    LOGGER.info("Evaluation plots written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
