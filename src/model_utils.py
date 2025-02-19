"""
model_utils.py: Funciones para evaluar modelos de machine learning.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, x_test, y_test, model_name):
    """
    Evalúa un modelo de machine learning y muestra métricas de rendimiento.

    Args:
        model: Modelo entrenado.
        X_test (pd.DataFrame): Conjunto de prueba de features.
        y_test (pd.Series): Conjunto de prueba de la variable objetivo.
        model_name (str): Nombre del modelo.

    Returns:
        None
    """
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nEvaluación del modelo {model_name}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
