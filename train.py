"""
train.py: Script para entrenar un modelo XGBoost y guardarlo en model.joblib.
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model_utils import evaluate_model

# Ruta de los datos preprocesados
DATA_PATH = "data/prep/train_clean.csv"
MODEL_PATH = "model.joblib"

def main():
    """Carga los datos, entrena el modelo y lo guarda."""
    # Cargar los datos preprocesados
    df = pd.read_csv(DATA_PATH)

    # Separar features y variable objetivo
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir y entrenar el modelo
    model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test, "XGBoost")

    # Guardar el modelo entrenado
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()
