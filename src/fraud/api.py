"""REST API that serves the claims fraud model.

Run locally:   uvicorn fraud.api:app --app-dir src --port 8000
Docker:        docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
Docs:          http://localhost:8000/docs
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud.data import CATEGORICAL, NUMERIC, engineer

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[2] / "models" / "fraud_model.joblib"))

# Raw fields the caller must send; engineered features are derived server-side.
RAW_REQUIRED = sorted(
    set(CATEGORICAL)
    | (set(NUMERIC) - {"policy_age_days", "vehicle_age", "injury_share", "property_share",
                        "vehicle_share", "claim_to_premium", "incident_weekday"})
    | {"policy_bind_date", "incident_date"}
)


class Claim(BaseModel):
    """A single claim in the raw schema of the source dataset."""
    features: dict[str, Any] = Field(..., description="Raw claim fields, e.g. incident_severity, total_claim_amount")


class Reason(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    fraud_score: float
    threshold: float
    decision: str
    top_reasons: list[Reason]
    model_version: str


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python src/fraud/train.py` first.")
    return joblib.load(MODEL_PATH)


def to_frame(raw: dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in RAW_REQUIRED if c not in raw]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_fields": missing})
    df = pd.DataFrame([raw]).replace("?", np.nan)
    return engineer(df)[CATEGORICAL + NUMERIC]


def explain(model, X: pd.DataFrame, k: int = 5) -> list[Reason]:
    """Top-k SHAP contributions (log-odds) for one claim."""
    import shap

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    xt = prep.transform(X)
    xt = xt.toarray() if hasattr(xt, "toarray") else xt
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    values = shap.TreeExplainer(clf)(xt).values[0]
    order = np.argsort(-np.abs(values))[:k]
    return [Reason(feature=names[i], contribution=round(float(values[i]), 4)) for i in order]


app = FastAPI(title="Claims Fraud Triage API", version="1.0.0",
              description="Scores an insurance claim for fraud risk and explains the score.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/schema")
def schema() -> dict:
    return {"required_fields": RAW_REQUIRED}


@app.post("/predict", response_model=Prediction)
def predict(claim: Claim) -> Prediction:
    bundle = load_bundle()
    model, threshold = bundle["model"], float(bundle["threshold"])
    X = to_frame(claim.features)
    score = float(model.predict_proba(X)[0, 1])
    return Prediction(
        fraud_score=round(score, 4),
        threshold=round(threshold, 4),
        decision="refer_to_siu" if score >= threshold else "fast_track",
        top_reasons=explain(model, X),
        model_version=type(model.named_steps["clf"]).__name__,
    )
