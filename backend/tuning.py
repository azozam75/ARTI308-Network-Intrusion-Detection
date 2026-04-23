"""Week-7 hyperparameter sweep for the IForest NIDS.

Grid over ``n_estimators`` × ``max_samples`` × ``max_features``. For each
cell: fit IForest on the BENIGN train pool, score val, pick the F1-optimal
threshold, and record binary + per-attack metrics. Results land in
``outputs/results/tuning_results.json``; the best config is then retrained
via ``backend.model.run_training`` so ``iforest.joblib`` and
``run_summary.json`` reflect the tuned bundle.

``contamination`` is intentionally **not** swept. sklearn applies it only
through ``offset_`` (``decision_function = score_samples - offset_``), so
across configs our ``-decision_function`` differs by a scalar offset. Since
the flagging threshold is selected post-hoc by F1 grid, the offset cancels
out — sweeping ``contamination`` cannot change F1, precision, recall, or
ROC-AUC in this setup.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.evaluation import evaluate_attack_wise, evaluate_binary
from backend.model import (
    ModelConfig,
    anomaly_scores,
    load_selected_features,
    load_split,
    run_training,
    select_threshold,
    set_seed,
    train_iforest,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Attack families where the baseline (Session 4) failed. Tracked separately
# in the sweep output so we can see whether any config recovers them.
WEAK_ATTACK_FAMILIES = ("PortScan", "Brute Force", "Botnet")


@dataclass
class TuneGrid:
    n_estimators: list[int] = field(default_factory=lambda: [100, 200, 400])
    # Per-tree sample draw. ``0.1`` = 10 % of the BENIGN pool (≈40 k with the
    # default ``train_subsample=400_000``) — materially larger per-tree views
    # than the sklearn default of 256.
    max_samples: list[int | float] = field(
        default_factory=lambda: [256, 1024, 4096, 0.1]
    )
    max_features: list[float] = field(default_factory=lambda: [0.5, 0.8, 1.0])


def iter_configs(base: ModelConfig, grid: TuneGrid) -> list[ModelConfig]:
    return [
        replace(base, n_estimators=ne, max_samples=ms, max_features=mf)
        for ne, ms, mf in itertools.product(
            grid.n_estimators, grid.max_samples, grid.max_features
        )
    ]


def config_key(cfg: ModelConfig) -> dict[str, int | float | str]:
    return {
        "n_estimators": cfg.n_estimators,
        "max_samples": cfg.max_samples,
        "max_features": cfg.max_features,
    }


def evaluate_config(
    config: ModelConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    set_seed(config.random_state)
    model = train_iforest(train, feature_cols, config)
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    scores = anomaly_scores(model, X_val)
    y_binary = (
        (val[config.category_column] != config.benign_label)
        .astype(np.int8)
        .to_numpy()
    )
    threshold, thr_stats = select_threshold(
        scores, y_binary, config.threshold_grid_size
    )
    preds = (scores >= threshold).astype(np.int8)
    binary = evaluate_binary(y_binary, preds, scores)
    attack = evaluate_attack_wise(
        val[config.category_column].to_numpy(),
        preds,
        scores,
        benign_label=config.benign_label,
    )
    weak_recall = {
        fam: attack.get(fam, {}).get("recall", 0.0) for fam in WEAK_ATTACK_FAMILIES
    }
    return {
        "config": config_key(config),
        "threshold": threshold,
        "threshold_stats": thr_stats,
        "val_metrics": binary,
        "val_attack_metrics": attack,
        "weak_family_recall": weak_recall,
    }


def pick_best(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["val_metrics"]["f1"])


def run_tuning(
    grid: TuneGrid | None = None,
    base_config: ModelConfig | None = None,
    retrain_best: bool = True,
) -> Path:
    grid = grid or TuneGrid()
    base = base_config or ModelConfig()
    base.outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = load_selected_features(base.selected_features_path)
    extras = [base.label_column, base.category_column]
    train = load_split(base.processed_dir, "train", feature_cols, extras)
    val = load_split(base.processed_dir, "val", feature_cols, extras)

    configs = iter_configs(base, grid)
    LOGGER.info("Evaluating %d configs", len(configs))

    results: list[dict] = []
    sweep_started = time.perf_counter()
    for i, cfg in enumerate(configs, 1):
        LOGGER.info("[%d/%d] %s", i, len(configs), config_key(cfg))
        t0 = time.perf_counter()
        record = evaluate_config(cfg, train, val, feature_cols)
        record["fit_seconds"] = time.perf_counter() - t0
        results.append(record)
        LOGGER.info(
            "  -> F1=%.4f ROC-AUC=%.4f FPR=%.4f weak-recall=%s (%.1fs)",
            record["val_metrics"]["f1"],
            record["val_metrics"]["roc_auc"],
            record["val_metrics"]["false_positive_rate"],
            {k: round(v, 3) for k, v in record["weak_family_recall"].items()},
            record["fit_seconds"],
        )

    best = pick_best(results)
    LOGGER.info(
        "Best config: %s  F1=%.4f  ROC-AUC=%.4f",
        best["config"],
        best["val_metrics"]["f1"],
        best["val_metrics"]["roc_auc"],
    )

    tuning_out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_wall_seconds": time.perf_counter() - sweep_started,
        "grid": asdict(grid),
        "n_features": len(feature_cols),
        "n_configs": len(results),
        "weak_family_names": list(WEAK_ATTACK_FAMILIES),
        "results": results,
        "best": best,
    }
    tuning_path = base.outputs_dir / "tuning_results.json"
    tuning_path.write_text(
        json.dumps(tuning_out, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", tuning_path)

    if retrain_best:
        best_cfg = replace(
            base,
            n_estimators=int(best["config"]["n_estimators"]),
            max_samples=best["config"]["max_samples"],
            max_features=float(best["config"]["max_features"]),
        )
        # run_training re-fits & persists iforest.joblib + run_summary.json
        # using the same pipeline as the baseline, so the bundle format stays
        # consistent for evaluate_test_split in Week 8.
        del train, val
        run_training(best_cfg)
        summary_path = base.outputs_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["selected_from_tuning"] = True
        summary["tuning_results_path"] = str(tuning_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        LOGGER.info("Tagged run_summary.json as tuned bundle")

    _print_leaderboard(results, top_n=5)
    return tuning_path


def _print_leaderboard(results: list[dict], top_n: int = 5) -> None:
    ranked = sorted(results, key=lambda r: r["val_metrics"]["f1"], reverse=True)
    print("\n=== Top configs by val F1 ===")
    header = (
        f"{'rank':>4}  {'n_est':>6}  {'max_samp':>10}  {'max_feat':>9}  "
        f"{'F1':>7}  {'ROC-AUC':>8}  {'FPR':>7}  "
        f"{'PortScan':>9}  {'BruteF':>7}  {'Botnet':>7}  {'time_s':>7}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked[:top_n], 1):
        c = r["config"]
        m = r["val_metrics"]
        w = r["weak_family_recall"]
        print(
            f"{i:>4}  {c['n_estimators']:>6}  {str(c['max_samples']):>10}  "
            f"{c['max_features']:>9}  {m['f1']:>7.4f}  {m['roc_auc']:>8.4f}  "
            f"{m['false_positive_rate']:>7.4f}  "
            f"{w['PortScan']:>9.4f}  {w['Brute Force']:>7.4f}  "
            f"{w['Botnet']:>7.4f}  {r['fit_seconds']:>7.1f}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_tuning()
