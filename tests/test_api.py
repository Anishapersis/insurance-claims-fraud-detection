"""API tests: run after `python src/fraud/train.py` has produced models/fraud_model.joblib."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from fraud import api  # noqa: E402
from fraud.data import load_raw  # noqa: E402

pytestmark = pytest.mark.skipif(not api.MODEL_PATH.exists(), reason="train the model first")
client = TestClient(api.app)


def sample_claim() -> dict:
    row = load_raw().iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_score_decision_and_reasons():
    r = client.post("/predict", json={"features": sample_claim()})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["decision"] in {"refer_to_siu", "fast_track"}
    assert len(body["top_reasons"]) == 5


def test_missing_fields_are_rejected():
    r = client.post("/predict", json={"features": {"age": 40}})
    assert r.status_code == 422
    assert "missing_fields" in r.json()["detail"]
