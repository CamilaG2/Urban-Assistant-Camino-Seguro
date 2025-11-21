# src/config.py
from pathlib import Path

# ----------------------------------------------
# RUTA DEL PROYECTO (carpeta raíz Proyecto_Final)
# ----------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------
# MODELO TFLITE (exportado desde Colab)
# ----------------------------------------------
TFLITE_MODEL_PATH = ROOT / "model" / "camino_seguro_v1" / "weights" / "best_saved_model" / "best_float32.tflite"

# ----------------------------------------------
# CLASES (según tu Roboflow)
# ----------------------------------------------
YOLO_CLASSES = ['Anden', 'Basura', 'Dog', 'Hueco', 'Objetos', 'Persona', 'Poste', 'Semaforo', 'Silla', 'Stop']

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.6

# ----------------------------------------------
# CÁMARA
# ----------------------------------------------
CAMERA_SOURCE = 0        # Webcam del PC
#CAMERA_SOURCE = "http://TU_IP_DEL_CELU:8080/video"       # Camara del celular revisa tu IP

# ----------------------------------------------
# PROFUNDIDAD / MiDaS
# ----------------------------------------------
CALIBRATION_FACTOR = 0.005   # luego se puede recalibrar
USE_DEPTH = True             # usamos MiDaS

# ----------------------------------------------
# ALERTAS (para alerts.py)
# ----------------------------------------------
# Tiempo mínimo entre alertas del mismo tipo (segundos)
DEBOUNCE_TIME = 0.8