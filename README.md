# 🚀 Urban Assistant – Camino Seguro

_Asistente de Visión por Computador para detección de obstáculos urbanos en tiempo real_

## 📌 Descripción general

Camino Seguro es un prototipo de asistencia urbana basado en visión por computador que detecta obstáculos en tiempo real y genera alertas auditivas para personas con discapacidad visual.

El sistema combina:

* YOLOv8 Small → TensorFlow Lite para detección ultra ligera en CPU

* MiDaS Small para estimación de profundidad monocular

* Lógica de priorización + Alertas por voz usando pyttsx3

* Arquitectura completa en Python, sin necesidad de GPU

* Su objetivo es servir como base para una futura implementación móvil en Android.

## 🧱 Estructura del Proyecto

```
    Proyecto_Final/
    │
    ├── data/
    │   ├── test/
    │   ├── train/
    |   ├── valid/
    |   ├── data.yaml
    |   ├── README.dataset.txt
    |   └── README.roboflow.txt
    |
    ├── model/
    │   └── camino_seguro_v1/
    │       └── weights/
    │           └── best_saved_model/
    │               └── best_float32.tflite   <-- Modelo TFLite
    |   ├── camino_seguro_val/
    |   └── inferencia_prueba  
    │
    ├── src/
    │   ├── __init__.py
    │   ├── alerts.py
    │   ├── config.py
    │   ├── main.py
    |   ├── models.py
    |   └── processing.py
    │
    ├── Documento.pdf
    ├── requirements.txt
    └── README.md

```
En la carpeta de model no solamente encontrarás tu modelo, también encontrarás el rendimiento, metricas y ajustes.

## 📦 Requisitos del sistema

1. Versión de Python Requerida

    Este proyecto fue desarrollado y probado en:

        ```
            Python 3.10.x
        ```
    No funcionará correctamente con Python 3.12 o superior debido a incompatibilidades con PyTorch, MiDaS y TensorFlow Lite.

2. Windows 10/11
3. CPU (no requiere GPU)
4. Cámara web

## ▶️ Cómo Ejecutar el Sistema

1. Descargar Dataset + Modelo Entrenado (TFLite)

    _Para ejecutar el proyecto necesitas descargar la carpeta data/ y la carpeta model/:_

    ### 👉 Descargar aquí (Google Drive):

        🔗 https://drive.google.com/drive/folders/1PPg40q32DGuAmBE8CZhlbE8wCzwRQbW3?usp=drive_link

        Colócalas en la raíz del proyecto de esta forma:

            ```
                Proyecto_Final/
                │── data/
                │── model/
                │── src/
                │── Documento.pdf
                │── README.md
                └── requirements.txt

            ```
2. Crea un entorno virtual y activalo
    ```
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1   
    ```
3. Ejecuta el archivo _requirements.txt_ en la raiz de tu carpeta
    ```
        pip install -r requirements.txt
    ```
4. Ejecuta el archivo _main.py_ en la raiz de tu carpeta
    ```
        python -m src.main
    ```
## 👩🏽 Autor
Autor: Maria Camila García Ramírez
Proyecto académico – Universidad del Rosario