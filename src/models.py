# src/models.py
import tensorflow as tf
import torch
from . import config


def load_tflite_model():
    """Carga modelo YOLO TFLite y devuelve (interpreter, input_details, output_details, INPUT_SHAPE)."""
    model_path = str(config.TFLITE_MODEL_PATH)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Asumimos entrada [1, H, W, 3]
    INPUT_SHAPE = input_details[0]["shape"][1:3]

    print(f"✅ YOLO TFLite cargado desde:\n{model_path}")
    print(f"   Entrada esperada: {INPUT_SHAPE}")

    return interpreter, input_details, output_details, INPUT_SHAPE


def load_midas_model():
    """Carga MiDaS (modelo pequeño) para estimar profundidad."""
    if not config.USE_DEPTH:
        print("ℹ️ USE_DEPTH=False, MiDaS no se cargará.")
        return None, None, None

    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)

        print(f"✅ MiDaS cargado en {device}")
        return midas, transform, device

    except Exception as e:
        print(f"❌ Error cargando MiDaS: {e}")
        return None, None, None