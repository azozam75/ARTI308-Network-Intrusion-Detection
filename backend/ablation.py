"""Week-9 per-category thresholding ablation.

For each attack category C, we fit a threshold on val that maximizes
F1(BENIGN vs C), then evaluate on test. The baseline comparison is the
global threshold from ``run_summary.json`` (same F1-grid procedure but
on BENIGN vs any-attack). Both methods are measured on the BENIGN ∪ C
test subset, so FPR is category-specific — this is deliberate, it
shows how much false-positive budget each attack would actually cost
under its own threshold.

This is an *oracle upper bound*: at inference time the true category
is unknown, so per-category thresholds cannot be applied directly.
The value of this ablation is interpretive — it quantifies how much
of the per-attack recall gap is due to the global-threshold assumption
vs. a genuine deficit in the IForest score signal.

Outputs:
  outputs/results/ablation_results.json      — full table + macro stats
  outputs/figures/per_category_ablation.png  — grouped-bar recall chart
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, f1_score, roc_curve

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
BENIGN_LABEL = "BENIGN"


def fit_category_threshold(
    scores: np.ndarray, y: np.ndarray, grid_size: int = 401
) -> tuple[float, float]:
    """Grid-search the threshold that maximizes F1 on a single BENIGN+C subset.

    Mirrors ``backend.model.select_threshold`` (percentile grid, skip
    degenerate predictions) so results are directly comparable to the
    global threshold.
    """
    qs = np.linspace(0.0, 1.0, grid_size)
    candidates = np.unique(np.quantile(scores, qs))
    best_f1 = -1.0
    best_thr = float(candidates[0])
    for thr in candidates:
        pred = (scores >= thr).astype(np.int8)
        n_pos = int(pred.sum())
        if n_pos == 0 or n_pos == len(pred):
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def _subset_metrics(
    scores: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def per_category_ablation(
    val: pd.DataFrame,
    test: pd.DataFrame,
    global_threshold: float,
    grid_size: int = 401,
) -> dict:
    categories = sorted(
        c for c in val["attack_category"].unique() if c != BENIGN_LABEL
    )
    records: dict[str, dict] = {}

    for cat in categories:
        val_sub = val.loc[val["attack_category"].isin([BENIGN_LABEL, cat])]
        test_sub = test.loc[test["attack_category"].isin([BENIGN_LABEL, cat])]
        y_val = (val_sub["attack_category"] == cat).astype(np.int8).to_numpy()
        y_test = (test_sub["attack_category"] == cat).astype(np.int8).to_numpy()
        s_val = val_sub["score"].to_numpy()
        s_test = test_sub["score"].to_numpy()

        per_thr, per_val_f1 = fit_category_threshold(s_val, y_val, grid_size)
        per_metrics = _subset_metrics(s_test, y_test, per_thr)
        glob_metrics = _subset_metrics(s_test, y_test, global_threshold)

        # Full ROC / AUC — diagnostic: signal-in-score-space vs threshold choice.
        fpr_curve, tpr_curve, _ = roc_curve(y_test, s_test)
        cat_auc = float(auc(fpr_curve, tpr_curve))

        records[cat] = {
            "n_test_attack": int(y_test.sum()),
            "n_test_benign": int((y_test == 0).sum()),
            "test_auc": cat_auc,
            "val_f1_at_per_category_threshold": per_val_f1,
            "global": glob_metrics,
            "per_category": per_metrics,
            "delta": {
                "recall": per_metrics["recall"] - glob_metrics["recall"],
                "f1": per_metrics["f1"] - glob_metrics["f1"],
                "fpr": per_metrics["false_positive_rate"]
                - glob_metrics["false_positive_rate"],
            },
        }

    def _macro(which: str) -> dict[str, float]:
        return {
            m: float(np.mean([records[c][which][m] for c in categories]))
            for m in ("precision", "recall", "f1", "false_positive_rate")
        }

    return {
        "global_threshold": float(global_threshold),
        "categories": records,
        "macro_global": _macro("global"),
        "macro_per_category": _macro("per_category"),
    }


def plot_per_category_recall(ablation: dict, out_path: Path) -> None:
    """Grouped bar chart: global-threshold recall vs per-category recall, per attack."""
    categories = list(ablation["categories"].keys())
    glob_recall = [ablation["categories"][c]["global"]["recall"] for c in categories]
    per_recall = [ablation["categories"][c]["per_category"]["recall"] for c in categories]
    per_fpr = [ablation["categories"][c]["per_category"]["false_positive_rate"] for c in categories]

    x = np.arange(len(categories))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(categories) + 3), 6))
    bars_g = ax.bar(
        x - width / 2, glob_recall, width,
        color="#4c72b0", label="Global threshold (baseline)",
    )
    bars_p = ax.bar(
        x + width / 2, per_recall, width,
        color="#dd8452", label="Per-category threshold (oracle)",
    )

    for bar, val in zip(bars_g, glob_recall):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar, val, fpr in zip(bars_p, per_recall, per_fpr):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}\n(FPR {fpr:.2f})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Recall on test")
    ax.set_title(
        "Per-category thresholding ablation — recall vs global baseline"
    )
    ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_table(ablation: dict) -> None:
    categories = list(ablation["categories"].keys())
    header = (
        f"{'category':<13}  {'AUC':>6}  "
        f"{'glob_recall':>11}  {'glob_fpr':>8}  "
        f"{'per_recall':>10}  {'per_fpr':>7}  {'d_recall':>9}"
    )
    print("\n=== Per-category thresholding ablation ===")
    print(header)
    print("-" * len(header))
    for cat in categories:
        r = ablation["categories"][cat]
        g, p = r["global"], r["per_category"]
        d = r["delta"]
        print(
            f"{cat:<13}  {r['test_auc']:>6.3f}  "
            f"{g['recall']:>11.4f}  {g['false_positive_rate']:>8.4f}  "
            f"{p['recall']:>10.4f}  {p['false_positive_rate']:>7.4f}  "
            f"{d['recall']:>+9.4f}"
        )
    mg, mp = ablation["macro_global"], ablation["macro_per_category"]
    print("-" * len(header))
    print(
        f"{'MACRO':<13}  {'':>6}  "
        f"{mg['recall']:>11.4f}  {mg['false_positive_rate']:>8.4f}  "
        f"{mp['recall']:>10.4f}  {mp['false_positive_rate']:>7.4f}  "
        f"{mp['recall'] - mg['recall']:>+9.4f}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text(encoding="utf-8"))
    global_threshold = float(summary["threshold"])

    LOGGER.info("Loading scored splits")
    val = pd.read_parquet(RESULTS_DIR / "val_scored.parquet")
    test = pd.read_parquet(RESULTS_DIR / "test_scored.parquet")

    LOGGER.info("Running per-category ablation")
    ablation = per_category_ablation(val, test, global_threshold)

    ablation_path = RESULTS_DIR / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(ablation, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", ablation_path)

    plot_path = FIGURES_DIR / "per_category_ablation.png"
    plot_per_category_recall(ablation, plot_path)
    LOGGER.info("Wrote %s", plot_path)

    _print_table(ablation)


if __name__ == "__main__":
    main()
