from fastapi import FastAPI, HTTPException
from typing import List, Any, Dict
import joblib
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "LogisticRegression.pkl")
MODEL_PATH = os.path.join("src", "models", MODEL_NAME)
PREPROC_PATH = os.path.join("src", "models", "preprocessor.pkl")

app = FastAPI(title="Telco Churn API")

# Variables globales para modelo y preprocesador
model = None
preprocessor = None


@app.on_event("startup")
def load_artifacts():
    """Carga el modelo y el preprocesador al arrancar la app."""
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
    """Asegura que el dataframe tenga todas las columnas esperadas.

    Añade NaN si faltan columnas.
    """
    for c in expected_columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[expected_columns]


@app.post("/predict")
def predict(payload: List[Dict[str, Any]]):
    """Espera una lista de registros JSON: [{col: val, ...}, ...]

    Acepta tanto Pydantic v1 como v2 ya que no se depende de RootModel.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessor not available"
        )

    data = payload
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(
            status_code=400,
            detail="Payload must be a non-empty list of records"
        )

    try:
        df = pd.DataFrame(data)

        # If preprocessor has attribute 'feature_names_in_', use it
        try:
            expected = list(preprocessor.feature_names_in_)
        except Exception:
            # Fallback: try to infer from provided data
            expected = list(df.columns)

        df = _ensure_columns(df, expected)

        X = preprocessor.transform(df)

        # If the transformer returns a numpy array, try to recover feature names
        X_for_model = X
        feature_names = None
        try:
            # sklearn compatible: prefer passing input feature names
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_names = list(
                        preprocessor.get_feature_names_out(df.columns)
                    )
                except Exception:
                    # fallback to no-arg get_feature_names_out
                    try:
                        feature_names = list(
                            preprocessor.get_feature_names_out()
                        )
                    except Exception:
                        feature_names = None
        except Exception:
            feature_names = None

        if feature_names is None:
            # fallback to attribute possibly set during preprocessing
            try:
                feature_names = list(preprocessor.feature_names_in_)
            except Exception:
                feature_names = None

        # If we have an ndarray and feature names, convert to DataFrame
        if isinstance(X, (np.ndarray,)) and feature_names is not None:
            try:
                if X.ndim == 2 and len(feature_names) == X.shape[1]:
                    X_for_model = pd.DataFrame(X, columns=feature_names)
                else:
                    # shape mismatch -> do not convert
                    X_for_model = X
            except Exception:
                X_for_model = X

        # As a final fallback, if the sklearn model stored feature_names_in_
        if isinstance(X_for_model, (np.ndarray,)):
            try:
                if hasattr(model, "feature_names_in_"):
                    feat = list(model.feature_names_in_)
                    if (X_for_model.ndim == 2 and
                            len(feat) == X_for_model.shape[1]):
                        X_for_model = pd.DataFrame(X_for_model, columns=feat)
            except Exception:
                pass

        # Ensure the DataFrame has exactly the same columns as the model
        try:
            if (hasattr(model, "feature_names_in_") and
                    isinstance(X_for_model, pd.DataFrame)):
                model_feats = list(model.feature_names_in_)
                # Reindex will add missing columns filled with 0
                X_for_model = X_for_model.reindex(
                    columns=model_feats, fill_value=0
                )
        except Exception:
            pass

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_for_model)[:, 1].tolist()
        else:
            # handle numpy arrays or DataFrames
            if hasattr(X_for_model, "shape"):
                probs = [None] * X_for_model.shape[0]
            else:
                probs = []

        preds = model.predict(X_for_model).tolist()
        # Map 1/0 to Yes/No if applicable
        mapped = ["Yes" if p == 1 or p == "1" else "No" for p in preds]

        return {"predictions": mapped, "probabilities": probs}

    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))
