import numpy as np
import pandas as pd

from backend.feature_selection import prune_correlated


def test_prune_correlated_drops_more_globally_redundant_member() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    shared = rng.normal(size=n)
    c_only = rng.normal(size=n)
    eps = 0.001
    # b and c are near-duplicates via `shared`, so (b, c) sits above threshold.
    # c ALSO partially reads `c_only`, which d is a noisy copy of. So c is
    # moderately correlated with d (below threshold) while b is not. The
    # heuristic should drop c — globally more redundant than b.
    b = shared + eps * rng.normal(size=n)
    c = shared + 0.2 * c_only + eps * rng.normal(size=n)
    d = c_only + 0.1 * rng.normal(size=n)
    a = rng.normal(size=n)  # independent sanity feature
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

    retained, dropped = prune_correlated(df, list(df.columns), threshold=0.95)
    dropped_names = {entry["dropped"] for entry in dropped}

    assert retained == ["a", "b", "d"]
    assert dropped_names == {"c"}
    assert dropped[0]["kept"] == "b"


def test_prune_correlated_respects_threshold() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x": rng.normal(size=2_000),
        "y": rng.normal(size=2_000),  # independent
    })
    retained, dropped = prune_correlated(df, ["x", "y"], threshold=0.95)
    assert retained == ["x", "y"]
    assert dropped == []
