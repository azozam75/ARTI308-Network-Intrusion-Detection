"""Metric suite for the IForest NIDS.

Two layers:
- ``evaluate_binary`` — BENIGN vs. any-attack detection metrics.
- ``evaluate_attack_wise`` — per-category recall + score distribution so
  we can see which attacks IForest catches vs. misses.

This module is imported by ``backend.model`` for val-time evaluation.
Its ``__main__`` runs the **one-shot** test evaluation: it loads the
saved model bundle and appends test metrics to ``run_summary.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_binary(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict:
    """Binary metrics with BENIGN=0, attack=1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {
            "benign": int((y_true == 0).sum()),
            "attack": int((y_true == 1).sum()),
        },
    }


def evaluate_attack_wise(
    y_true_category: np.ndarray,
    y_pred_binary: np.ndarray,
    scores: np.ndarray,
    benign_label: str = "BENIGN",
) -> dict:
    """Per-category detection rate + score distribution stats.

    For non-BENIGN categories the reported rate is recall (fraction of
    attack rows flagged). For BENIGN it is the false-positive rate.
    """
    categories = pd.Series(y_true_category).unique().tolist()
    out: dict[str, dict] = {}
    for cat in sorted(categories):
        mask = y_true_category == cat
        n = int(mask.sum())
        if n == 0:
            continue
        cat_scores = scores[mask]
        cat_pred = y_pred_binary[mask]
        flagged = float((cat_pred == 1).mean())
        rate_key = "false_positive_rate" if cat == benign_label else "recall"
        out[cat] = {
            "n": n,
            rate_key: flagged,
            "score_mean": float(cat_scores.mean()),
            "score_median": float(np.median(cat_scores)),
            "score_std": float(cat_scores.std()),
            "score_p10": float(np.quantile(cat_scores, 0.1)),
            "score_p90": float(np.quantile(cat_scores, 0.9)),
        }
    return out


def score_features(
    model, features: pd.DataFrame, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = -model.decision_function(features.to_numpy(dtype=np.float32))
    preds = (scores >= threshold).astype(np.int8)
    return scores, preds


def evaluate_test_split(
    processed_dir: Path,
    outputs_dir: Path,
    category_column: str = "AttackCategory",
    benign_label: str = "BENIGN",
    label_column: str = "Label",
) -> dict:
    """One-shot test evaluation — run only after hyperparameter tuning is done.

    Reads ``iforest.joblib`` (bundle: model + threshold + feature_cols),
    scores the test parquet, writes ``test_scored.parquet`` and merges
    ``test_metrics`` + ``test_attack_metrics`` into ``run_summary.json``.
    """
    bundle = joblib.load(outputs_dir / "iforest.joblib")
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]

    test = pd.read_parquet(
        processed_dir / "test.parquet",
        columns=feature_cols + [label_column, category_column],
    )
    LOGGER.info("Scoring test set (%d rows)", len(test))
    scores, preds = score_features(model, test[feature_cols], threshold)
    y_binary = (
        (test[category_column] != benign_label).astype(np.int8).to_numpy()
    )

    binary = evaluate_binary(y_binary, preds, scores)
    attack = evaluate_attack_wise(
        test[category_column].to_numpy(), preds, scores, benign_label=benign_label
    )

    pd.DataFrame({
        "score": scores.astype(np.float32),
        "prediction": preds,
        "label": test[label_column].to_numpy(),
        "attack_category": test[category_column].to_numpy(),
    }).to_parquet(outputs_dir / "test_scored.parquet", index=False)

    summary_path = outputs_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_metrics"] = binary
    summary["test_attack_metrics"] = attack
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote test metrics to %s", summary_path)

    print("\n=== Test metrics ===")
    print(f"  Accuracy  : {binary['accuracy']:.4f}")
    print(f"  Precision : {binary['precision']:.4f}")
    print(f"  Recall    : {binary['recall']:.4f}")
    print(f"  F1        : {binary['f1']:.4f}")
    print(f"  ROC-AUC   : {binary['roc_auc']:.4f}")
    print(f"  FPR       : {binary['false_positive_rate']:.4f}")
    print("\n=== Per-attack detection (test) ===")
    for cat, stats in attack.items():
        rate_key = "false_positive_rate" if cat == benign_label else "recall"
        print(f"  {cat:<14} n={stats['n']:>7}  {rate_key}={stats[rate_key]:.4f}")

    return {"binary": binary, "attack_wise": attack}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    processed = PROJECT_ROOT / "data" / "processed"
    outputs = PROJECT_ROOT / "outputs" / "results"
    evaluate_test_split(processed, outputs)
