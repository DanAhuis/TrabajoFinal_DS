"""
Módulo: aggregation.py
Provee funciones para explorar y resumir el dataset.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def summarize_data(df: pd.DataFrame) -> None:
    """
    Muestra un resumen del dataset:
    - Tipos de datos
    - Valores nulos
    - Estadísticas básicas
    """
    try:
        logger.info("=== RESUMEN DE DATOS ===")
        logger.info(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        logger.info("\nTipos de datos:\n" + str(df.dtypes))
        logger.info("\nValores nulos por columna:\n" + str(df.isnull().sum()))
        logger.info(
            "\nEstadísticas descriptivas:\n" +
            str(df.describe(include="all").transpose())
        )
    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        raise
