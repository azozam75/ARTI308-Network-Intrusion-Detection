"""Week-2 EDA on the processed train split.

Produces:
  outputs/figures/class_balance.png    — attack category counts (log y)
  outputs/figures/correlation_heatmap.png — feature correlation matrix
  outputs/results/eda_summary.txt      — shape, class counts, NaN check
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def plot_class_balance(train: pd.DataFrame, out_path: Path) -> None:
    counts = train["AttackCategory"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Rows (log scale)")
    ax.set_xlabel("")
    ax.set_title("CIC-IDS-2017 — train split class balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(train: pd.DataFrame, features: list[str], out_path: Path) -> None:
    # Sample for speed; 70×70 on 2M rows is overkill.
    sample = train[features].sample(n=min(200_000, len(train)), random_state=42)
    corr = sample.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation (train, 200k sample)")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(train: pd.DataFrame, features: list[str], out_path: Path) -> None:
    counts = train["AttackCategory"].value_counts()
    lines = [
        f"Train shape: {train.shape}",
        f"Feature count: {len(features)}",
        f"NaNs in features: {int(train[features].isna().sum().sum())}",
        "",
        "Class distribution:",
        counts.to_string(),
        "",
        "Imbalance ratio (majority / minority): "
        f"{counts.max() / counts.min():,.1f}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((PROCESSED_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    features: list[str] = manifest["feature_columns"]

    LOGGER.info("Loading train split")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    LOGGER.info("Class balance plot")
    plot_class_balance(train, FIGURES_DIR / "class_balance.png")

    LOGGER.info("Correlation heatmap")
    plot_correlation_heatmap(train, features, FIGURES_DIR / "correlation_heatmap.png")

    LOGGER.info("Summary")
    write_summary(train, features, RESULTS_DIR / "eda_summary.txt")

    LOGGER.info("EDA artifacts in %s and %s", FIGURES_DIR, RESULTS_DIR)


if __name__ == "__main__":
    main()
