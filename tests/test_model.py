import numpy as np
import pandas as pd

from backend.model import (
    ModelConfig,
    build_model,
    select_threshold,
    train_iforest,
)


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.random_state == 42
    assert cfg.benign_label == "BENIGN"
    assert cfg.n_estimators == 200
    assert cfg.category_column == "AttackCategory"


def test_train_iforest_fits_only_on_benign_rows() -> None:
    rng = np.random.default_rng(0)
    n = 1_500
    df = pd.DataFrame({
        "f0": rng.normal(size=n).astype(np.float32),
        "f1": rng.normal(size=n).astype(np.float32),
        "AttackCategory": rng.choice(["BENIGN", "DoS"], size=n, p=[0.7, 0.3]),
        "Label": "x",
    })
    cfg = ModelConfig(n_estimators=10, train_subsample=None, n_jobs=1)
    actual = train_iforest(df, ["f0", "f1"], cfg)

    benign_only = (
        df.loc[df["AttackCategory"] == "BENIGN", ["f0", "f1"]]
        .to_numpy(dtype=np.float32)
    )
    reference = build_model(cfg).fit(benign_only)

    probe = df[["f0", "f1"]].head(80).to_numpy(dtype=np.float32)
    np.testing.assert_allclose(
        actual.decision_function(probe),
        reference.decision_function(probe),
    )


def test_select_threshold_separates_two_populations() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=600)
    anomalous = rng.normal(loc=4.0, scale=1.0, size=150)
    scores = np.concatenate([normal, anomalous])
    y = np.concatenate([np.zeros(600), np.ones(150)]).astype(np.int8)

    thr, stats = select_threshold(scores, y, grid_size=201)

    assert stats["f1"] > 0.85
    assert 1.0 < thr < 4.0


def test_select_threshold_skips_degenerate_predictions() -> None:
    scores = np.linspace(0.0, 1.0, 200)
    y = (scores > 0.5).astype(np.int8)
    thr, stats = select_threshold(scores, y, grid_size=51)
    preds = (scores >= thr).astype(np.int8)
    assert 0 < preds.sum() < len(preds)
    assert stats["f1"] > 0.95
