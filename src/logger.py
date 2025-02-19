"""
logger.py: Configuración del sistema de logging.
"""

import logging
import os
import yaml

# Cargar configuración desde YAML
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

LOG_FILE = config["paths"]["logs"]
LOG_DIR = os.path.dirname(LOG_FILE)  # Obtener la ruta del directorio

# Crear la carpeta de logs si no existe
os.makedirs(LOG_DIR, exist_ok=True)

# Configuración del logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)


def get_logger():
    """
    Retorna un logger configurado.

    Returns:
        logging.Logger: Logger configurado.
    """
    return logging.getLogger()
