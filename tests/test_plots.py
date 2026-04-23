from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from backend.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_score_distributions,
)


def _synthetic_scored(seed: int, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_benign = int(n * 0.7)
    n_dos = (n - n_benign) // 2
    n_portscan = n - n_benign - n_dos
    scores = np.concatenate([
        rng.normal(-0.1, 0.05, size=n_benign),
        rng.normal(0.1, 0.10, size=n_dos),
        rng.normal(-0.08, 0.04, size=n_portscan),
    ]).astype(np.float32)
    categories = (
        ["BENIGN"] * n_benign + ["DoS"] * n_dos + ["PortScan"] * n_portscan
    )
    preds = (scores >= 0.0).astype(np.int8)
    return pd.DataFrame({
        "score": scores,
        "prediction": preds,
        "label": categories,
        "attack_category": categories,
    })


def test_plot_roc_curves_writes_png_and_returns_aucs(tmp_path: Path) -> None:
    val = _synthetic_scored(seed=0)
    test = _synthetic_scored(seed=1)
    out = tmp_path / "roc.png"

    stats = plot_roc_curves(val, test, threshold=0.0, out_path=out)

    assert out.exists() and out.stat().st_size > 0
    assert 0.0 <= stats["val_auc"] <= 1.0
    assert 0.0 <= stats["test_auc"] <= 1.0


def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    test = _synthetic_scored(seed=2)
    out = tmp_path / "cm.png"

    plot_confusion_matrix(test, threshold=0.0, out_path=out)

    assert out.exists() and out.stat().st_size > 0


def test_plot_score_distributions_writes_png(tmp_path: Path) -> None:
    test = _synthetic_scored(seed=3)
    out = tmp_path / "dist.png"

    plot_score_distributions(test, threshold=0.0, out_path=out)

    assert out.exists() and out.stat().st_size > 0
