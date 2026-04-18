"""CIC-IDS-2017 preprocessing — raw CSVs → train/val/test parquet + fitted preprocessor + manifest."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Infiltration (~36 rows) and Heartbleed (~11 rows) are out of scope per the
# project proposal and dropped. Web Attack labels contain U+FFFD
# (replacement char) in the source CSVs where an en-dash was corrupted
# during dataset preparation; matched verbatim below.
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack \ufffd Brute Force": "Web Attack",
    "Web Attack \ufffd XSS": "Web Attack",
    "Web Attack \ufffd Sql Injection": "Web Attack",
    "Bot": "Botnet",
}


@dataclass
class DataConfig:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    file_encoding: str = "utf-8"
    label_column: str = "Label"
    category_column: str = "AttackCategory"
    attack_category_map: dict[str, str] = field(
        default_factory=lambda: dict(ATTACK_CATEGORY_MAP)
    )
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    impute_strategy: str = "median"
    clip_quantile_low: float = 0.001
    clip_quantile_high: float = 0.999
    drop_zero_variance: bool = True


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to learned [low_q, high_q] bounds.

    Fit on train only so inference-time clipping uses train-derived bounds
    (no leakage). Applied after imputation but before scaling so extreme
    tails don't inflate the StandardScaler's mean/std.
    """

    def __init__(self, low: float = 0.001, high: float = 0.999) -> None:
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        self.low_ = np.nanquantile(arr, self.low, axis=0)
        self.high_ = np.nanquantile(arr, self.high, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.clip(arr, self.low_, self.high_)


def load_raw_csvs(
    raw_dir: Path, encoding: str
) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    LOGGER.info("Loading %d CSVs from %s", len(files), raw_dir)
    parts: list[pd.DataFrame] = []
    for path in files:
        part = pd.read_csv(path, encoding=encoding, low_memory=False)
        numeric_cols = part.select_dtypes(include="number").columns
        # float32 cast halves peak memory during concat. Flow Duration
        # (microseconds, up to ~120M) exceeds float32's contiguous-integer
        # range (2^24 ≈ 16M); downstream clip+scale absorbs the rounding.
        part[numeric_cols] = part[numeric_cols].astype(np.float32)
        LOGGER.info("  %s -> %s", path.name, part.shape)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    parts.clear()
    LOGGER.info("Combined raw shape: %s", combined.shape)
    return combined, files


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace; drop duplicate `Fwd Header Length`."""
    df = df.rename(columns=lambda c: c.strip())
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        LOGGER.info(
            "Dropping duplicate columns: %s",
            df.columns[dup_mask].tolist(),
        )
        df = df.loc[:, ~dup_mask].copy()
    return df


def validate_schema(df: pd.DataFrame, label_col: str) -> None:
    if label_col not in df.columns:
        raise ValueError(
            f"Expected label column {label_col!r} not found. "
            f"Columns: {df.columns.tolist()}"
        )
    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")


def map_attack_categories(
    df: pd.DataFrame,
    label_col: str,
    category_col: str,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Map fine labels to coarse categories; drop out-of-scope rows."""
    before = len(df)
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df[category_col] = df[label_col].map(mapping)
    unmapped_mask = df[category_col].isna()
    if unmapped_mask.any():
        counts = df.loc[unmapped_mask, label_col].value_counts()
        LOGGER.info("Dropping out-of-scope labels:\n%s", counts.to_string())
    df = df.loc[~unmapped_mask].reset_index(drop=True)
    LOGGER.info("Rows %d -> %d after category mapping", before, len(df))
    LOGGER.info(
        "Category distribution:\n%s",
        df[category_col].value_counts().to_string(),
    )
    return df


def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN — arises from zero-duration flows in Flow Bytes/s etc."""
    numeric_cols = df.select_dtypes(include="number").columns
    block = df[numeric_cols].to_numpy(copy=False)
    mask = np.isinf(block)
    inf_count = int(mask.sum())
    if inf_count:
        df[numeric_cols] = df[numeric_cols].mask(mask, np.nan)
        LOGGER.info("Replaced %d infinite values with NaN", inf_count)
    return df


def drop_zero_variance_features(
    df: pd.DataFrame, protect: Iterable[str]
) -> pd.DataFrame:
    protect_set = set(protect)
    candidates = [c for c in df.columns if c not in protect_set]
    nunique = df[candidates].nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        LOGGER.info(
            "Dropping %d zero-variance columns: %s", len(zero_var), zero_var
        )
        df = df.drop(columns=zero_var)
    return df


def split_data(
    df: pd.DataFrame,
    stratify_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[stratify_col],
        random_state=random_state,
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        LOGGER.info("%s shape: %s", name, part.shape)
        LOGGER.info(
            "%s class distribution:\n%s",
            name,
            part[stratify_col].value_counts().to_string(),
        )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_preprocessor(config: DataConfig) -> Pipeline:
    """Imputer → quantile clip → StandardScaler (fit on train only)."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy=config.impute_strategy)),
            (
                "clip",
                QuantileClipper(
                    low=config.clip_quantile_low,
                    high=config.clip_quantile_high,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )


def apply_preprocessor(
    df: pd.DataFrame, preprocessor: Pipeline, feature_cols: list[str]
) -> pd.DataFrame:
    """Transform feature columns in place; returned for chaining."""
    df[feature_cols] = preprocessor.transform(df[feature_cols]).astype(np.float32)
    return df


def save_artifacts(
    splits: dict[str, pd.DataFrame],
    preprocessor: Pipeline,
    feature_cols: list[str],
    config: DataConfig,
    source_files: list[Path],
) -> Path:
    out_dir = config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        LOGGER.info("Wrote %s (%s)", path, part.shape)
    preprocessor_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    LOGGER.info("Wrote %s", preprocessor_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in source_files],
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "splits": {name: list(part.shape) for name, part in splits.items()},
        "feature_columns": feature_cols,
        "preprocessor_path": str(preprocessor_path),
    }
    manifest_path = out_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", manifest_path)
    return out_dir


def build_dataset(
    config: DataConfig,
) -> tuple[pd.DataFrame, list[str], list[Path]]:
    df, source_files = load_raw_csvs(config.raw_dir, config.file_encoding)
    df = clean_column_names(df)
    validate_schema(df, config.label_column)
    df = map_attack_categories(
        df,
        config.label_column,
        config.category_column,
        config.attack_category_map,
    )
    df = replace_infinities(df)
    if config.drop_zero_variance:
        df = drop_zero_variance_features(
            df, protect=[config.label_column, config.category_column]
        )
    feature_cols = [
        c for c in df.columns
        if c not in {config.label_column, config.category_column}
    ]
    LOGGER.info(
        "Clean dataset: %s, %d feature columns", df.shape, len(feature_cols)
    )
    return df, feature_cols, source_files


def run_pipeline(config: DataConfig | None = None) -> Path:
    """Full pipeline: load → clean → split → fit → transform → save."""
    config = config or DataConfig()
    df, feature_cols, source_files = build_dataset(config)
    train, val, test = split_data(
        df,
        stratify_col=config.category_column,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )
    preprocessor = build_preprocessor(config)
    preprocessor.fit(train[feature_cols])
    splits = {"train": train, "val": val, "test": test}
    for part in splits.values():
        apply_preprocessor(part, preprocessor, feature_cols)
    out_dir = save_artifacts(
        splits=splits,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        config=config,
        source_files=source_files,
    )
    LOGGER.info("Pipeline complete. Artifacts in %s", out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline()
