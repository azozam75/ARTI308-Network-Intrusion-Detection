import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.preprocessing import (
    ATTACK_CATEGORY_MAP,
    DataConfig,
    QuantileClipper,
    clean_column_names,
    map_attack_categories,
    run_pipeline,
)


def test_clean_column_names_strips_and_dedupes() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=[" Fwd Header Length", "Fwd Header Length", " Other "])
    out = clean_column_names(df)
    assert list(out.columns) == ["Fwd Header Length", "Other"]


def test_map_attack_categories_drops_out_of_scope() -> None:
    df = pd.DataFrame({
        "Label": ["BENIGN", "DoS Hulk", "Heartbleed", "Infiltration", "Bot"],
    })
    out = map_attack_categories(df, "Label", "AttackCategory", dict(ATTACK_CATEGORY_MAP))
    assert len(out) == 3
    assert set(out["AttackCategory"]) == {"BENIGN", "DoS", "Botnet"}
    assert out["AttackCategory"].isna().sum() == 0


def test_map_attack_categories_preserves_web_attack_with_fffd() -> None:
    # Source CSVs contain U+FFFD between "Attack" and the subtype where an
    # en-dash was corrupted. Guard against silent drop if the map drifts.
    df = pd.DataFrame({
        "Label": [
            "Web Attack \ufffd Brute Force",
            "Web Attack \ufffd XSS",
            "Web Attack \ufffd Sql Injection",
        ],
    })
    out = map_attack_categories(df, "Label", "AttackCategory", dict(ATTACK_CATEGORY_MAP))
    assert len(out) == 3
    assert (out["AttackCategory"] == "Web Attack").all()


def test_quantile_clipper_clips_to_learned_bounds() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10_000, 3))
    clipper = QuantileClipper(low=0.01, high=0.99)
    clipper.fit(X)
    Xt = clipper.transform(X)
    assert (Xt >= clipper.low_).all()
    assert (Xt <= clipper.high_).all()
    # Bounds learned from the same data should clip ~2% of values.
    assert (Xt != X).mean() < 0.05


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(42)
    n = 400

    data = {
        " Destination Port": rng.integers(0, 65535, n).astype(np.int64),
        " Flow Duration": rng.integers(1, 10_000_000, n).astype(np.int64),
        " Flow Bytes/s": np.where(rng.random(n) < 0.05, np.inf, rng.random(n) * 1000),
        " Constant Column": np.zeros(n),
        " Label": rng.choice(
            ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "Heartbleed"], size=n
        ),
    }
    pd.DataFrame(data).to_csv(raw / "day1.csv", index=False)

    config = DataConfig(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        test_size=0.2,
        val_size=0.2,
        random_state=0,
    )
    out_dir = run_pipeline(config)

    for name in ("train", "val", "test"):
        assert (out_dir / f"{name}.parquet").exists()
    assert (out_dir / "preprocessor.joblib").exists()

    manifest = json.loads((out_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert "Constant Column" not in manifest["feature_columns"]

    train = pd.read_parquet(out_dir / "train.parquet")
    assert "Heartbleed" not in train["Label"].unique()
    assert set(train["AttackCategory"].unique()) <= {"BENIGN", "DoS", "DDoS", "PortScan"}
    assert np.isfinite(train[manifest["feature_columns"]].to_numpy()).all()
