import numpy as np
import pandas as pd

from backend.model import ModelConfig
from backend.tuning import (
    TuneGrid,
    WEAK_ATTACK_FAMILIES,
    evaluate_config,
    iter_configs,
    pick_best,
)


def _synthetic_split(n: int, seed: int, benign_frac: float = 0.7) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    n_benign = int(n * benign_frac)
    n_attack = n - n_benign
    benign = rng.normal(0.0, 1.0, size=(n_benign, 4)).astype(np.float32)
    attack = rng.normal(3.0, 1.0, size=(n_attack, 4)).astype(np.float32)
    X = np.vstack([benign, attack])
    cols = [f"f{i}" for i in range(4)]
    df = pd.DataFrame(X, columns=cols)
    df["AttackCategory"] = ["BENIGN"] * n_benign + ["DoS"] * n_attack
    df["Label"] = df["AttackCategory"]
    return df, cols


def test_iter_configs_covers_full_cartesian_grid() -> None:
    grid = TuneGrid(
        n_estimators=[50, 100],
        max_samples=[64, 128],
        max_features=[0.5, 1.0],
    )
    configs = iter_configs(ModelConfig(), grid)

    assert len(configs) == 8
    triples = {(c.n_estimators, c.max_samples, c.max_features) for c in configs}
    assert len(triples) == 8
    assert {c.random_state for c in configs} == {42}


def test_evaluate_config_returns_finite_metrics_on_synthetic() -> None:
    train, cols = _synthetic_split(n=1500, seed=0)
    val, _ = _synthetic_split(n=800, seed=1)
    cfg = ModelConfig(
        n_estimators=20,
        max_samples=64,
        train_subsample=None,
        threshold_grid_size=51,
        n_jobs=1,
    )

    record = evaluate_config(cfg, train, val, cols)

    m = record["val_metrics"]
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert record["config"]["n_estimators"] == 20
    assert "DoS" in record["val_attack_metrics"]
    # Weak-family keys are always present for stable downstream schema even
    # when the synthetic split doesn't contain those categories.
    assert set(record["weak_family_recall"]) == set(WEAK_ATTACK_FAMILIES)


def test_pick_best_selects_highest_val_f1() -> None:
    results = [
        {"config": {"n_estimators": 50}, "val_metrics": {"f1": 0.30}},
        {"config": {"n_estimators": 100}, "val_metrics": {"f1": 0.72}},
        {"config": {"n_estimators": 200}, "val_metrics": {"f1": 0.55}},
    ]
    best = pick_best(results)
    assert best["config"]["n_estimators"] == 100
