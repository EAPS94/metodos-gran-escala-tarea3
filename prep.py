"""
prep.py: Script para preprocesar los datos y guardarlos en data/prep/
"""

import pandas as pd
import os
from src.data_utils import clean_data, encode_categorical, select_features

# Ruta de los datos
DATA_PATH = "data/raw/train.csv"
OUTPUT_PATH = "data/prep/train_clean.csv"

def main():
    """Carga, limpia, selecciona características y guarda los datos preprocesados."""
    df = pd.read_csv(DATA_PATH)

    # Aplicar limpieza de datos
    df_cleaned = clean_data(df)

    # Convertir variables categóricas a numéricas
    df_encoded = encode_categorical(df_cleaned)

    # Selección de características
    df_final = select_features(df_encoded)

    # Guardar datos preprocesados
    os.makedirs("data/prep", exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Datos preprocesados guardados en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
