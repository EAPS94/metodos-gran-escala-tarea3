"""
predict.py: Funciones para realizar inferencias con un modelo entrenado.
"""

import os
import sys  # Módulos estándar primero
import pandas as pd
import yaml  # Librerías de terceros después

from src.logger import get_logger  # Finalmente, módulos locales

# Asegura que 'src' esté en el path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Cargar configuración desde YAML
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

logger = get_logger()
validation_rules = config["validation"]


def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que los valores en el DataFrame
    estén dentro de los rangos permitidos.

    Args:
        df (pd.DataFrame): Datos de entrada.

    Returns:
        pd.DataFrame: Datos validados.
    """
    errors = []
    for column, (min_val, max_val) in validation_rules.items():
        if column in df.columns:
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            if not out_of_range.empty:
                errors.append(
                    f"Valores fuera de rango en {column}: "
                    f"{out_of_range[column].values}")

    if errors:
        logger.error("Errores de validación: %s", errors)
        raise ValueError(f"Errores de validación: {errors}")

    return df


def make_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa un modelo entrenado para hacer predicciones
    en un nuevo conjunto de datos.

    Args:
        model: Modelo de machine learning cargado.
        df (pd.DataFrame): Datos de entrada para predicción.

    Returns:
        pd.DataFrame: DataFrame con las predicciones.
    """
    try:
        df = validate_inputs(df)
        expected_features = model.feature_names_in_
        df = df[expected_features]
        predictions = model.predict(df)
        result_df = df.copy()
        result_df["Predicted_SalePrice"] = predictions
        return result_df
    except Exception as e:
        logger.error("Error en la predicción: %s", e)
        raise
