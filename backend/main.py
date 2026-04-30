"""FastAPI routes for the SENTINEL dashboard.

Serves the ML artefacts produced by Weeks 5-9 so the React frontend can
render live results without shipping JSON/PNGs into the build. Also
exposes a lightweight scoring endpoint (`/predict`) that loads the
saved IsolationForest bundle once at startup and returns the anomaly
score + flag for a preprocessed feature vector.
"""

from __future__ import annotations

import json
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _allowed_figures() -> set[str]:
    if not FIGURES_DIR.exists():
        return set()
    return {p.name for p in FIGURES_DIR.glob("*.png")}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained IForest bundle once at startup.

    We tolerate a missing bundle so the API still serves JSON/figures in
    environments where the model hasn't been trained locally (e.g. CI).
    """
    bundle_path = RESULTS_DIR / "iforest.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        app.state.model = bundle["model"]
        app.state.threshold = float(bundle["threshold"])
        app.state.feature_cols = list(bundle["feature_cols"])
    else:
        app.state.model = None
        app.state.threshold = None
        app.state.feature_cols = None
    yield


app = FastAPI(
    title="SENTINEL NIDS API",
    description="Isolation Forest NIDS — ARTI 308 Group 2 · IAU",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., description="Preprocessed feature vector in selected-features order."
    )


class PredictResponse(BaseModel):
    score: float
    flagged: bool
    threshold: float


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-nids"}


@app.get("/metrics")
def metrics() -> Any:
    return _load_json(RESULTS_DIR / "run_summary.json")


@app.get("/tuning")
def tuning() -> Any:
    return _load_json(RESULTS_DIR / "tuning_results.json")


@app.get("/ablation")
def ablation() -> Any:
    return _load_json(RESULTS_DIR / "ablation_results.json")


@app.get("/features")
def features() -> Any:
    return _load_json(PROCESSED_DIR / "selected_features.json")


@app.get("/figures")
def list_figures() -> dict[str, list[str]]:
    return {"figures": sorted(_allowed_figures())}


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    # Whitelist-check to block path traversal; only files that actually
    # exist in FIGURES_DIR are served.
    if name not in _allowed_figures():
        raise HTTPException(status_code=404, detail=f"figure '{name}' not found")
    return FileResponse(FIGURES_DIR / name, media_type="image/png")


@app.get("/sample")
def sample() -> dict[str, Any]:
    """Return a random pre-scored row from the test split for UI demos."""
    test_scored = RESULTS_DIR / "test_scored.parquet"
    if not test_scored.exists():
        raise HTTPException(status_code=404, detail="test_scored.parquet not found")
    df = pd.read_parquet(test_scored)
    idx = random.randrange(len(df))
    row = df.iloc[idx]
    return {
        "index": int(idx),
        "score": float(row["score"]),
        "prediction": int(row["prediction"]),
        "label": str(row["label"]),
        "attack_category": str(row["attack_category"]),
        "threshold": app.state.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle not loaded — train via `python -m backend.model`.",
        )
    expected = len(app.state.feature_cols)
    if len(req.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"expected {expected} features in selected-features order, "
                f"got {len(req.features)}"
            ),
        )
    X = np.asarray([req.features], dtype=np.float32)
    # Mirror `backend.model.anomaly_scores` — higher = more anomalous.
    score = float(-app.state.model.decision_function(X)[0])
    return PredictResponse(
        score=score,
        flagged=score >= app.state.threshold,
        threshold=app.state.threshold,
    )
