# src/main.py

import cv2
from . import config
from . import models
from . import processing
from . import alerts   # <--- añadimos alerts


def format_distance(distance):
    """Redondea distancia a pasos de 0.5 m para que no salte tanto."""
    if distance is None:
        return "N/A"
    d = round(distance * 2) / 2.0
    return f"{d:.1f} m"


def process_frame(frame, interpreter_data, midas_data):
    detections = processing.get_yolo_detections(frame, interpreter_data)
    depth_map = processing.get_depth_map(frame, midas_data)

    output = frame.copy()
    H, W, _ = frame.shape

    for (x1, y1, x2, y2, cls, conf) in detections:
        distance = processing.calculate_distance([x1, y1, x2, y2], depth_map)

        box_h = y2 - y1
        box_ratio = box_h / float(H)

        print(
            f"[DEBUG] {cls} conf={conf:.2f} "
            f"dist={'None' if distance is None else f'{distance:.2f}'} "
            f"ratio={box_ratio:.2f}"
        )

        dist_txt = format_distance(distance)

        # ================== LÓGICA DE VOZ ==================
        # Si tenemos distancia numérica usamos eso
        if distance is not None:
            # alerta si está a menos de 3 m
            if distance < 3.0:
                alerts.alert(f"{cls} a {dist_txt}", cls)
        else:
            # Sin profundidad: usamos tamaño de la caja
            if cls == "Persona":
                if box_ratio > 0.30:
                    alerts.alert("Persona muy cerca", cls)
            else:
                if box_ratio > 0.40:
                    alerts.alert(f"{cls} muy cerca", cls)
        # ===================================================

        color = (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{cls} {dist_txt}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output


def main():
    interpreter_data = models.load_tflite_model()
    midas_data = models.load_midas_model()

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    cv2.namedWindow("Urban Assistant", cv2.WINDOW_NORMAL)

    print("🎥 Corriendo… presiona Q (o ESC) en la ventana de video para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer frame de la cámara.")
            continue

        frame_proc = process_frame(frame, interpreter_data, midas_data)

        cv2.imshow("Urban Assistant", frame_proc)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("👋 Tecla de salida detectada, cerrando…")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()