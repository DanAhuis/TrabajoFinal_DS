from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "LogisticRegression.pkl")
MODEL_PATH = os.path.join("src", "models", MODEL_NAME)
PREPROC_PATH = os.path.join("src", "models", "preprocessor.pkl")

app = FastAPI(title="Telco Churn API")


class Payload(BaseModel):
    __root__: List[Dict[str, Any]]


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.warning(f"No se pudo cargar modelo en {MODEL_PATH}: {e}")
        model = None

    try:
        preprocessor = joblib.load(PREPROC_PATH)
    except Exception as e:
        logger.warning(f"No se pudo cargar preprocesador en {PREPROC_PATH}: {e}")
        preprocessor = None


@app.get("/health")
def health():
    return {"status": "ok"}


def _ensure_columns(df: pd.DataFrame, expected_columns: List[str]):
    """Asegura que el dataframe tenga todas las columnas esperadas (añade NaN si faltan)."""
    for c in expected_columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[expected_columns]


@app.post("/predict")
def predict(payload: Payload):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not available")

    data = payload.__root__
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty list of records")

    try:
        df = pd.DataFrame(data)

        # If preprocessor has attribute 'feature_names_in_', use it to ensure columns
        try:
            expected = list(preprocessor.feature_names_in_)
        except Exception:
            # Fallback: try to infer from training data in data/processed if available
            expected = list(df.columns)

        df = _ensure_columns(df, expected)

        X = preprocessor.transform(df)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1].tolist()
        else:
            probs = [None] * X.shape[0]

        preds = model.predict(X).tolist()
        # Map 1/0 to Yes/No if applicable
        mapped = ["Yes" if p == 1 or p == "1" else "No" for p in preds]

        return {"predictions": mapped, "probabilities": probs}

    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))
