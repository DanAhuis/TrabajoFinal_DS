import pytest
import os

# Skip if fastapi is not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import app, load_artifacts  # noqa: E402

# Cargar los artefactos antes de crear el cliente de prueba
load_artifacts()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(
    not os.path.exists("src/models/preprocessor.pkl"),
    reason="No preprocessor available"
)
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
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        print(f"Response: {r.text}")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert "probabilities" in body


def test_list_models():
    """Prueba el endpoint para listar modelos disponibles."""
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert "preprocessor" in body
    assert "current_model" in body
    assert "metrics_available" in body


def test_train_models():
    """Prueba el endpoint de entrenamiento de modelos."""
    r = client.post("/train")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "message" in body
    assert "duration_seconds" in body
    assert "models_available" in body
    assert "LogisticRegression.pkl" in body["models_available"]
