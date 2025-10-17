import pytest
import os

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(not os.path.exists("src/models/preprocessor.pkl"), reason="No preprocessor available")
def test_predict_single():
    sample = [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
            "Churn": "No",
        }
    ]
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert "probabilities" in body
