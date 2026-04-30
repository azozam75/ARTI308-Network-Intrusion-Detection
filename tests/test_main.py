"""Smoke tests for the FastAPI routes in ``backend.main``.

Uses FastAPI's ``TestClient`` so the startup/shutdown hooks fire (the
IForest bundle is loaded under ``app.state`` in the lifespan). Tests
are read-only and assume the Week-5 training artefacts exist on disk.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_has_val_and_test(client):
    data = client.get("/metrics").json()
    assert "val_metrics" in data
    assert "test_metrics" in data
    assert data["selected_from_tuning"] is True
    assert 0.0 < data["val_metrics"]["f1"] < 1.0


def test_ablation_has_all_categories(client):
    data = client.get("/ablation").json()
    expected = {"Botnet", "Brute Force", "DDoS", "DoS", "PortScan", "Web Attack"}
    assert expected.issubset(data["categories"].keys())


def test_figures_listing_matches_single_figure_fetch(client):
    names = client.get("/figures").json()["figures"]
    assert "confusion_matrix.png" in names
    r = client.get("/figures/confusion_matrix.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000


def test_figures_rejects_unknown_and_traversal(client):
    assert client.get("/figures/no_such.png").status_code == 404
    # Whitelist-check defeats ".." traversal — only exact filenames pass.
    assert client.get("/figures/..%2Fsecret.png").status_code == 404


def test_predict_roundtrip(client):
    feats = client.get("/features").json()["retained_features"]
    r = client.post("/predict", json={"features": [0.0] * len(feats)})
    assert r.status_code == 200
    body = r.json()
    assert "score" in body and "flagged" in body
    assert body["threshold"] == pytest.approx(-0.0397, abs=1e-3)


def test_predict_rejects_wrong_length(client):
    r = client.post("/predict", json={"features": [0.0] * 3})
    assert r.status_code == 422
    assert "expected" in r.json()["detail"]


def test_sample_returns_valid_row(client):
    body = client.get("/sample").json()
    assert body["prediction"] in (0, 1)
    assert body["attack_category"]  # non-empty string
