"""Correlation-based feature pruning on the processed train split.

Greedy drop: for every pair with |r| >= threshold, remove the member with
lower variance (less informative). Writes the retained list and the drop
log to data/processed/ for downstream model code to consume.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CORR_THRESHOLD = 0.95
SAMPLE_ROWS = 300_000
RANDOM_STATE = 42


def prune_correlated(
    df: pd.DataFrame, features: list[str], threshold: float
) -> tuple[list[str], list[dict]]:
    """Return (retained, dropped) where dropped[i] = {feature, paired_with, corr}."""
    corr = df[features].corr().abs()
    variances = df[features].var()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    dropped_set: set[str] = set()
    drop_log: list[dict] = []

    # Iterate pairs in descending correlation so strongest redundancies go first.
    pairs = (
        upper.stack()
        .loc[lambda s: s >= threshold]
        .sort_values(ascending=False)
    )
    for (a, b), r in pairs.items():
        if a in dropped_set or b in dropped_set:
            continue
        loser = a if variances[a] < variances[b] else b
        keeper = b if loser == a else a
        dropped_set.add(loser)
        drop_log.append({"dropped": loser, "kept": keeper, "abs_corr": float(r)})

    retained = [f for f in features if f not in dropped_set]
    return retained, drop_log


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = json.loads((PROCESSED_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    features: list[str] = manifest["feature_columns"]

    LOGGER.info("Loading train sample (%d rows)", SAMPLE_ROWS)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=features)
    sample = train.sample(n=min(SAMPLE_ROWS, len(train)), random_state=RANDOM_STATE)

    LOGGER.info("Pruning |r| >= %.2f across %d features", CORR_THRESHOLD, len(features))
    retained, drop_log = prune_correlated(sample, features, CORR_THRESHOLD)

    out = {
        "threshold": CORR_THRESHOLD,
        "sample_rows": len(sample),
        "n_input": len(features),
        "n_retained": len(retained),
        "n_dropped": len(drop_log),
        "retained_features": retained,
        "dropped": drop_log,
    }
    out_path = PROCESSED_DIR / "selected_features.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    LOGGER.info("Kept %d / %d features (dropped %d)", len(retained), len(features), len(drop_log))
    for entry in drop_log:
        LOGGER.info("  drop %-35s (|r|=%.3f with %s)", entry["dropped"], entry["abs_corr"], entry["kept"])
    LOGGER.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
