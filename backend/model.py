"""Semi-supervised Isolation Forest for anomaly-based NIDS.

Trained on BENIGN-only rows from the processed train split using the 50
features retained after correlation pruning. At inference, higher
``-decision_function`` ⇒ more anomalous. The flagging threshold is chosen
on the val set by maximizing binary F1 (BENIGN vs. any attack). The test
split is **not** touched here — it is reserved for the one-shot Week 8
evaluation in ``backend.evaluation``.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from backend.evaluation import evaluate_attack_wise, evaluate_binary

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    outputs_dir: Path = PROJECT_ROOT / "outputs" / "results"
    selected_features_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "selected_features.json"
    )
    label_column: str = "Label"
    category_column: str = "AttackCategory"
    benign_label: str = "BENIGN"
    n_estimators: int = 200
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = -1
    # Cap BENIGN train rows to keep fit wall-time reasonable. IsolationForest
    # internally subsamples `max_samples` per tree anyway, so this mostly
    # affects the pool from which each tree draws.
    train_subsample: int | None = 400_000
    threshold_grid_size: int = 401


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_selected_features(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["retained_features"])


def load_split(
    processed_dir: Path,
    name: str,
    feature_cols: list[str],
    extra_cols: list[str],
) -> pd.DataFrame:
    cols = feature_cols + extra_cols
    path = processed_dir / f"{name}.parquet"
    df = pd.read_parquet(path, columns=cols)
    LOGGER.info("Loaded %s: %s", path.name, df.shape)
    return df


def build_model(config: ModelConfig) -> IsolationForest:
    return IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        contamination=config.contamination,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (sklearn's convention is the opposite)."""
    return -model.decision_function(X)


def train_iforest(
    train: pd.DataFrame, feature_cols: list[str], config: ModelConfig
) -> IsolationForest:
    benign = train.loc[train[config.category_column] == config.benign_label]
    LOGGER.info("BENIGN rows in train: %d", len(benign))
    if config.train_subsample and len(benign) > config.train_subsample:
        benign = benign.sample(
            n=config.train_subsample, random_state=config.random_state
        )
        LOGGER.info("Subsampled BENIGN to %d for training", len(benign))
    X = benign[feature_cols].to_numpy(dtype=np.float32)
    model = build_model(config)
    LOGGER.info(
        "Fitting IsolationForest: n_estimators=%d, max_samples=%s, n=%d",
        config.n_estimators, config.max_samples, len(X),
    )
    model.fit(X)
    return model


def select_threshold(
    scores: np.ndarray, y_binary: np.ndarray, grid_size: int
) -> tuple[float, dict[str, float]]:
    """Grid-search the threshold that maximizes binary F1 on val.

    Grid is percentile-spaced over the score distribution — avoids the
    O(n log n) of sweeping every unique value on 400k+ rows without
    losing granularity around the decision boundary.
    """
    qs = np.linspace(0.0, 1.0, grid_size)
    candidates = np.unique(np.quantile(scores, qs))
    best_f1 = -1.0
    best_thr = float(candidates[0])
    best_stats: dict[str, float] = {}
    for thr in candidates:
        y_pred = (scores >= thr).astype(np.int8)
        n_positive = int(y_pred.sum())
        if n_positive == 0 or n_positive == len(y_pred):
            continue
        f1 = f1_score(y_binary, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = {
                "f1": float(f1),
                "precision": float(precision_score(y_binary, y_pred, zero_division=0)),
                "recall": float(recall_score(y_binary, y_pred, zero_division=0)),
                "positive_rate": float(n_positive / len(y_pred)),
            }
    LOGGER.info(
        "Selected threshold %.6f (F1=%.4f, P=%.4f, R=%.4f)",
        best_thr,
        best_stats.get("f1", 0.0),
        best_stats.get("precision", 0.0),
        best_stats.get("recall", 0.0),
    )
    return best_thr, best_stats


def run_training(config: ModelConfig | None = None) -> Path:
    config = config or ModelConfig()
    set_seed(config.random_state)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = load_selected_features(config.selected_features_path)
    LOGGER.info("Using %d selected features", len(feature_cols))

    extras = [config.label_column, config.category_column]
    train = load_split(config.processed_dir, "train", feature_cols, extras)
    val = load_split(config.processed_dir, "val", feature_cols, extras)

    model = train_iforest(train, feature_cols, config)
    del train

    LOGGER.info("Scoring val set (%d rows)", len(val))
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    scores_val = anomaly_scores(model, X_val)
    y_val_binary = (
        (val[config.category_column] != config.benign_label).astype(np.int8).to_numpy()
    )

    threshold, thr_stats = select_threshold(
        scores_val, y_val_binary, config.threshold_grid_size
    )
    y_val_pred = (scores_val >= threshold).astype(np.int8)

    binary_metrics = evaluate_binary(y_val_binary, y_val_pred, scores_val)
    attack_metrics = evaluate_attack_wise(
        y_true_category=val[config.category_column].to_numpy(),
        y_pred_binary=y_val_pred,
        scores=scores_val,
        benign_label=config.benign_label,
    )

    model_path = config.outputs_dir / "iforest.joblib"
    joblib.dump(
        {"model": model, "threshold": threshold, "feature_cols": feature_cols},
        model_path,
    )
    LOGGER.info("Wrote %s", model_path)

    val_scored_path = config.outputs_dir / "val_scored.parquet"
    pd.DataFrame({
        "score": scores_val.astype(np.float32),
        "prediction": y_val_pred,
        "label": val[config.label_column].to_numpy(),
        "attack_category": val[config.category_column].to_numpy(),
    }).to_parquet(val_scored_path, index=False)
    LOGGER.info("Wrote %s", val_scored_path)

    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "n_features": len(feature_cols),
        "threshold": threshold,
        "threshold_selection_stats": thr_stats,
        "val_metrics": binary_metrics,
        "val_attack_metrics": attack_metrics,
        "model_path": str(model_path),
    }
    summary_path = config.outputs_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", summary_path)

    print("\n=== Validation metrics ===")
    print(f"  Accuracy  : {binary_metrics['accuracy']:.4f}")
    print(f"  Precision : {binary_metrics['precision']:.4f}")
    print(f"  Recall    : {binary_metrics['recall']:.4f}")
    print(f"  F1        : {binary_metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {binary_metrics['roc_auc']:.4f}")
    print(f"  FPR       : {binary_metrics['false_positive_rate']:.4f}")
    print(f"  Threshold : {threshold:.6f}")
    print("\n=== Per-attack detection (val) ===")
    for cat, stats in attack_metrics.items():
        rate_key = "false_positive_rate" if cat == config.benign_label else "recall"
        print(f"  {cat:<14} n={stats['n']:>7}  {rate_key}={stats[rate_key]:.4f}")

    return summary_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_training()
