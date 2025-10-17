import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from loguru import logger

logger.add("logs/training.log", rotation="1 MB", level="INFO")

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "telco_prepared.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Entrena y evalúa un modelo, devolviendo sus métricas."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    return model, metrics


def main():
    logger.info("=== INICIANDO ENTRENAMIENTO DE MODELOS ===")

    # === 1. CARGAR DATOS ===
    df = pd.read_parquet(DATA_PROCESSED)
    logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 🔧 Convertir columna objetivo a binaria
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    # === 2. SEPARAR X e Y ===
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # === 3. TRAIN-TEST SPLIT ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Datos divididos: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # === 4. ENTRENAMIENTO ===
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    results = []

    for name, model in models.items():
        logger.info(f"Entrenando modelo: {name}")
        trained_model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)

        # Guardar modelo entrenado
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(trained_model, model_path)
        logger.info(f"Modelo guardado: {model_path}")

    # === 5. GUARDAR MÉTRICAS ===
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, "metrics.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Métricas guardadas en: {results_path}")
    logger.info("=== ENTRENAMIENTO FINALIZADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
