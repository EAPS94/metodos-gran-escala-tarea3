# Predicción del Precio de Casas

## Descripción
Este proyecto implementa un pipeline de **Machine Learning** para predecir el precio de venta de casas utilizando `XGBoost`. Sigue las mejores prácticas de **código limpio** y **producción** con logs, validaciones y configuración en `config.yaml`.

## Estructura del Repositorio
```
metodos-gran-escala-tarea3/
│── config.yaml                  # Configuración del proyecto
│── model.joblib                  # Modelo entrenado
│── README.md                     # Documentación del proyecto
│── prep.py                        # Preprocesamiento de datos
│── train.py                       # Entrenamiento del modelo
│── inference.py                   # Predicción desde archivo o entrada manual
│
├── data/
│   ├── raw/                      # Datos originales (train.csv)
│   ├── prep/                     # Datos preprocesados (train_clean.csv)
│   ├── inference/                 # Datos de inferencia (inference_data.csv)
│   └── predictions/               # Predicciones generadas
│
├── src/
│   ├── __init__.py               # Archivo para definir src como un módulo
│   ├── data_utils.py             # Funciones de preprocesamiento
│   ├── predict.py                 # Funciones de predicción
│   └── logger.py                 # Configuración de logs
│
└── logs/
    └── inference.log              # Logs de predicciones
```

## Instalación
1. Clona el repositorio:
```sh
git clone https://github.com/tu_usuario/metodos-gran-escala-tarea3.git
cd metodos-gran-escala-tarea3
```
2. Instala dependencias:
```sh
pip install -r requirements.txt
```
3. Ejecuta los scripts en este orden:
```sh
python prep.py   # Preprocesamiento de datos
python train.py  # Entrenamiento del modelo
python inference.py  # Inferencia (archivo o entrada manual)
```

## Uso
### **Inferencia desde archivo**
```sh
python inference.py
```
**Selecciona** `archivo` e ingresa el dataset `data/inference/inference_data.csv`.

### **Inferencia manual**
```sh
python inference.py
```
**Selecciona** `manual` e ingresa los valores interactivos.


## Pruebas
Para verificar los logs generados:
```sh
Get-Content logs/inference.log
```

---

