""" 
inference.py: Script para cargar un modelo entrenado y hacer predicciones desde un archivo o entrada manual.
"""

import pandas as pd
import joblib
import os
import yaml
from src.predict import make_predictions, validate_inputs
from src.logger import get_logger

# Cargar configuración desde YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

logger = get_logger()

MODEL_PATH = config["paths"]["model"]
INPUT_DATA_PATH = config["paths"]["inference_data"]
OUTPUT_PREDICTIONS_PATH = config["paths"]["output_predictions"]

def manual_input_prediction(model):
    """
    Permite ingresar datos manualmente en la terminal para obtener una predicción.
    """
    try:
        print("\n🔹 Ingrese los datos de la casa para predecir su precio:")
        logger.info("Inicio de entrada manual para predicción.")

        user_input = {}
        validation_rules = config["validation"]

        for feature, (min_val, max_val) in validation_rules.items():
            while True:
                try:
                    value = float(input(f"Ingrese {feature} ({min_val}-{max_val}): "))
                    if min_val <= value <= max_val:
                        user_input[feature] = value
                        break
                    else:
                        print(f"⚠️ Valor fuera de rango. Debe estar entre {min_val} y {max_val}.")
                        logger.warning(f"Usuario ingresó un valor fuera de rango en {feature}: {value}")
                except ValueError:
                    print("⚠️ Entrada inválida. Por favor, ingrese un número.")
                    logger.warning(f"Entrada inválida en {feature}, usuario ingresó un valor no numérico.")

        # Convertir a DataFrame y validar
        input_data = pd.DataFrame([user_input])
        validate_inputs(input_data)

        # Hacer la predicción
        predicted_price = model.predict(input_data)[0]
        print(f"\n💰 **Precio estimado de la casa: ${predicted_price:,.2f}**")
        logger.info(f"Predicción manual generada con éxito: ${predicted_price:,.2f}")

    except Exception as e:
        logger.error(f"Error en entrada manual: {e}")
        print(f"❌ Error: {e}")

def main():
    """Carga el modelo y realiza predicciones desde un archivo o entrada manual."""
    try:
        # Cargar el modelo entrenado
        logger.info("Intentando cargar el modelo.")
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modelo cargado desde {MODEL_PATH}")
        logger.info(f"Modelo cargado correctamente desde {MODEL_PATH}")

        mode = input("\n¿Quieres hacer una predicción desde un archivo o ingresar datos manualmente? (archivo/manual): ").strip().lower()

        if mode == "archivo":
            logger.info("Modo seleccionado: Archivo.")
            df = pd.read_csv(INPUT_DATA_PATH)
            print(f"📂 Datos de inferencia cargados desde {INPUT_DATA_PATH}")
            logger.info(f"Datos de inferencia cargados desde {INPUT_DATA_PATH}")

            # Validar datos
            df = validate_inputs(df)

            # Hacer predicciones
            logger.info("Generando predicciones para el archivo de entrada.")
            predictions = make_predictions(model, df)

            # Guardar predicciones
            os.makedirs("data/predictions", exist_ok=True)
            predictions.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
            print(f"✅ Predicciones guardadas en {OUTPUT_PREDICTIONS_PATH}")
            logger.info(f"Predicciones guardadas en {OUTPUT_PREDICTIONS_PATH}")

        elif mode == "manual":
            logger.info("Modo seleccionado: Entrada manual.")
            manual_input_prediction(model)

        else:
            print("⚠️ Opción no válida. Ingresa 'archivo' o 'manual'.")
            logger.warning(f"Opción no válida ingresada: {mode}")

    except Exception as e:
        logger.error(f"Error en la ejecución de inference.py: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
