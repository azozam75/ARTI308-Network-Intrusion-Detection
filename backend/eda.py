"""Week-2 EDA on the processed train split.

Produces:
  outputs/figures/class_balance.png    — attack category counts (log y)
  outputs/figures/correlation_heatmap.png — feature correlation matrix
  outputs/figures/pca_scatter.png      — 2D PCA projection coloured by class
  outputs/results/eda_summary.txt      — shape, class counts, NaN check,
                                         PCA variance explained
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

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


def plot_pca_scatter(
    train: pd.DataFrame, features: list[str], out_path: Path
) -> tuple[float, float]:
    """Stratified 2D PCA projection; returns (PC1, PC2) variance explained."""
    per_class = 2_000
    sample = pd.concat(
        [
            g.sample(n=min(per_class, len(g)), random_state=42)
            for _, g in train.groupby("AttackCategory", observed=True)
        ],
        ignore_index=True,
    )
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(sample[features].to_numpy())
    sample = sample.assign(PC1=projected[:, 0], PC2=projected[:, 1])

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=sample["AttackCategory"].nunique())
    sns.scatterplot(
        data=sample, x="PC1", y="PC2", hue="AttackCategory",
        palette=palette, s=10, alpha=0.6, linewidth=0, ax=ax,
    )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% var)")
    ax.set_title("PCA(2) — stratified 2k-per-class sample (visual separability)")
    ax.legend(loc="best", markerscale=1.5, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return float(var[0]), float(var[1])


def write_summary(
    train: pd.DataFrame,
    features: list[str],
    pca_variance: tuple[float, float],
    out_path: Path,
) -> None:
    counts = train["AttackCategory"].value_counts()
    pc1, pc2 = pca_variance
    lines = [
        f"Train shape: {train.shape}",
        f"Feature count: {len(features)}",
        f"NaNs in features: {int(train[features].isna().sum().sum())}",
        "",
        "Class distribution:",
        counts.to_string(),
        "",
        f"Imbalance ratio (majority / minority): {counts.max() / counts.min():,.1f}",
        "",
        f"PCA variance explained — PC1: {pc1 * 100:.2f}%, PC2: {pc2 * 100:.2f}%, "
        f"combined: {(pc1 + pc2) * 100:.2f}%",
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

    LOGGER.info("PCA scatter")
    pca_var = plot_pca_scatter(train, features, FIGURES_DIR / "pca_scatter.png")

    LOGGER.info("Summary")
    write_summary(train, features, pca_var, RESULTS_DIR / "eda_summary.txt")

    LOGGER.info("EDA artifacts in %s and %s", FIGURES_DIR, RESULTS_DIR)


if __name__ == "__main__":
    main()
