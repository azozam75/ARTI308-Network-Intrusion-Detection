from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from backend.ablation import (
    fit_category_threshold,
    per_category_ablation,
    plot_per_category_recall,
)


def _synthetic_scored(
    seed: int,
    n_benign: int = 500,
    n_easy: int = 120,
    n_hard: int = 120,
) -> pd.DataFrame:
    """BENIGN ~ N(-0.1, 0.04), Easy attack ~ N(0.15, 0.06), Hard attack ~ N(-0.07, 0.04).

    Easy attack is cleanly separable, hard attack overlaps BENIGN heavily.
    """
    rng = np.random.default_rng(seed)
    scores = np.concatenate([
        rng.normal(-0.1, 0.04, size=n_benign),
        rng.normal(0.15, 0.06, size=n_easy),
        rng.normal(-0.07, 0.04, size=n_hard),
    ]).astype(np.float32)
    cats = (
        [BENIGN_LABEL] * n_benign + ["Easy"] * n_easy + ["Hard"] * n_hard
    )
    return pd.DataFrame({
        "score": scores,
        "prediction": 0,
        "label": cats,
        "attack_category": cats,
    })


BENIGN_LABEL = "BENIGN"


def test_fit_category_threshold_on_separable_data() -> None:
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, size=600)
    attack = rng.normal(4.0, 1.0, size=200)
    scores = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(600), np.ones(200)]).astype(np.int8)

    thr, f1 = fit_category_threshold(scores, y, grid_size=201)

    assert 1.0 < thr < 4.0
    assert f1 > 0.85


def test_per_category_ablation_beats_baseline_on_hard_category() -> None:
    val = _synthetic_scored(seed=0)
    test = _synthetic_scored(seed=1)
    # Set global threshold near the midpoint of BENIGN and Easy — it will
    # catch Easy cleanly but miss Hard entirely.
    global_threshold = 0.02

    ab = per_category_ablation(val, test, global_threshold, grid_size=201)

    assert set(ab["categories"]) == {"Easy", "Hard"}
    hard = ab["categories"]["Hard"]
    # Per-category threshold must recover *some* recall on Hard where the
    # global threshold scores ~0, even if the cost is higher FPR.
    assert hard["per_category"]["recall"] > hard["global"]["recall"]
    # Per-category macro recall should not be below global macro recall —
    # the ablation only ever improves or matches on F1-optimal thresholds.
    assert ab["macro_per_category"]["recall"] >= ab["macro_global"]["recall"]


def test_plot_per_category_recall_writes_png(tmp_path: Path) -> None:
    val = _synthetic_scored(seed=2)
    test = _synthetic_scored(seed=3)
    ab = per_category_ablation(val, test, 0.0, grid_size=101)
    out = tmp_path / "ablation.png"

    plot_per_category_recall(ab, out)

    assert out.exists() and out.stat().st_size > 0
