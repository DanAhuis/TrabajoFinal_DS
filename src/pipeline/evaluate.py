import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

# === CONFIGURACIÓN ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "telco_prepared.parquet")
MODELS_PATH = os.path.join(BASE_DIR, "src", "models")

def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa un modelo y muestra métricas y gráficos."""
    # 🔧 Asegurar que las etiquetas sean numéricas (0 = No, 1 = Yes)
    if y_test.dtype == 'object':
        y_test = y_test.map({'No': 0, 'Yes': 1})

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logging.info(f"\n=== Resultados {model_name} ===")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall: {rec:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    logging.info(f"ROC-AUC: {roc_auc:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    # Matriz de confusión
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()

    # Curva ROC
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f'Curva ROC - {model_name}')
    plt.show()

def main():
    logging.info("=== INICIANDO EVALUACIÓN DE MODELOS ===")

    # Cargar datos procesados
    df = pd.read_parquet(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Cargar modelos entrenados
    modelos = {
        "LogisticRegression": "LogisticRegression.pkl",
        "RandomForest": "RandomForest.pkl",
        "XGBoost": "XGBoost.pkl"
    }

    for nombre, archivo in modelos.items():
        ruta = os.path.join(MODELS_PATH, archivo)
        if os.path.exists(ruta):
            model = joblib.load(ruta)
            logging.info(f"Evaluando modelo: {nombre}")
            evaluate_model(model, X, y, nombre)
        else:
            logging.warning(f"No se encontró el modelo: {archivo}")

    logging.info("=== EVALUACIÓN FINALIZADA ===")

if __name__ == "__main__":
    main()
