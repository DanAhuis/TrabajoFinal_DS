"""
Módulo: cleaning.py
Limpieza inicial del dataset Telco Customer Churn.
Convierte tipos de datos, elimina valores faltantes y columnas irrelevantes.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Carga y limpia el dataset Telco Customer Churn.
    - Convierte columnas numéricas.
    - Elimina filas vacías o inconsistentes.
    - Quita columnas no útiles (como customerID).
    """

    try:
        # Cargar CSV
        df = pd.read_csv(filepath)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Convertir TotalCharges a numérico
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Eliminar filas con TotalCharges vacío
        df = df.dropna(subset=['TotalCharges'])

        # Eliminar identificadores irrelevantes
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        # Resetear índices
        df = df.reset_index(drop=True)

        logger.info(f"Datos limpiados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df

    except Exception as e:
        logger.error(f"Error durante la limpieza: {e}")
        raise
