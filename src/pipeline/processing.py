"""
Pipeline principal para el proyecto Telco Customer Churn.
Ejecuta los pasos de limpieza, codificación y preparación de datos.
Genera archivos intermedios en formato parquet.
"""

import joblib
import pandas as pd
import sys
import os
import logging


# =============================
# CONFIGURACIÓN DE LOGGING
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Agregar src al path (para acceder a preprocess)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importar funciones modulares
from preprocess.cleaning import load_and_clean_data
from preprocess.preprocessing import encode_and_scale
from preprocess.aggregation import summarize_data

# =============================
# FUNCIONES DE UTILIDAD
# =============================

def load_data(filepath: str) -> pd.DataFrame:
    """Carga datos desde archivo CSV."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        raise


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Guarda los datos procesados en formato parquet."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Datos guardados en: {filepath}")
    except Exception as e:
        logger.error(f"Error guardando datos: {e}")
        raise


# =============================
# PIPELINE PRINCIPAL
# =============================

def run_pipeline():
    """Ejecuta la pipeline completa para el dataset de churn."""
    logger.info("=== INICIANDO PIPELINE CHURN ===")

    # Definir rutas
    base_path = os.path.dirname(__file__)         # pipeline/
    project_root = os.path.dirname(base_path)     # src/
    data_root = os.path.dirname(project_root)     # raíz del proyecto

    input_path = os.path.join(data_root, "data", "Telco-Customer-Churn-dirty.csv")
    processed_dir = os.path.join(data_root, "data", "processed")

    if not os.path.exists(input_path):
        logger.error(f"No se encuentra el archivo: {input_path}")
        return

    try:
        # 1. CARGA DE DATOS
        df_raw = load_data(input_path)

               # 2. LIMPIEZA
        logger.info("Paso 1: Limpieza de datos")
        df_clean = load_and_clean_data(input_path)
        save_processed_data(df_clean, os.path.join(processed_dir, "telco_cleaned.parquet"))

        # 🧹 LIMPIEZA EXTRA: asegurar que MonthlyCharges sea numérico
        logger.info("Verificando columna 'MonthlyCharges' antes de limpiar:")
        logger.info(df_clean['MonthlyCharges'].value_counts().head(10))

        df_clean['MonthlyCharges'] = pd.to_numeric(df_clean['MonthlyCharges'], errors='coerce')
        df_clean = df_clean.dropna(subset=['MonthlyCharges'])

        logger.info("Verificando columna 'MonthlyCharges' después de limpiar:")
        logger.info(df_clean['MonthlyCharges'].describe())

        # 3. ANÁLISIS DESCRIPTIVO
        logger.info("Paso 2: Resumen de datos")
        summarize_data(df_clean)

        # 4. PREPROCESAMIENTO (codificación + escalado)
        logger.info("Paso 3: Codificación y escalado")
        X_scaled, y, preprocessor, feature_names = encode_and_scale(df_clean)
        df_final = pd.DataFrame(X_scaled, columns=feature_names)
        df_final["Churn"] = y.values

        # 5. GUARDAR RESULTADO FINAL
        save_processed_data(df_final, os.path.join(processed_dir, "telco_prepared.parquet"))

        # Guardar el preprocesador completo
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocesador guardado en: {preprocessor_path}")


    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
