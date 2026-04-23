import numpy as np

from backend.evaluation import evaluate_attack_wise, evaluate_binary


def test_evaluate_binary_perfect_classifier() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    m = evaluate_binary(y_true, y_pred, scores)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}
    assert m["false_positive_rate"] == 0.0
    assert m["true_positive_rate"] == 1.0


def test_evaluate_binary_all_negative_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    m = evaluate_binary(y_true, y_pred, scores)
    assert m["recall"] == 0.0
    # zero_division=0: precision undefined → reported as 0 rather than NaN.
    assert m["precision"] == 0.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}
    assert m["support"] == {"benign": 2, "attack": 2}


def test_evaluate_attack_wise_reports_recall_and_fpr() -> None:
    y_cat = np.array([
        "BENIGN", "BENIGN", "BENIGN",
        "DoS", "DoS",
        "Web Attack",
    ])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.85])

    out = evaluate_attack_wise(y_cat, y_pred, scores)

    assert out["BENIGN"]["n"] == 3
    assert abs(out["BENIGN"]["false_positive_rate"] - 1 / 3) < 1e-9
    assert "recall" not in out["BENIGN"]

    assert out["DoS"]["n"] == 2
    assert out["DoS"]["recall"] == 0.5
    assert out["Web Attack"]["n"] == 1
    assert out["Web Attack"]["recall"] == 1.0
