# src/processing.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from . import config


# ---------------------- YOLO TFLite ---------------------- #

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(dets, iou_thresh=0.4):
    """
    Aplica NMS por clase sobre lista de dets:
    [x1,y1,x2,y2,class_name,conf]
    """
    if not dets:
        return dets

    dets = sorted(dets, key=lambda d: d[5], reverse=True)
    final = []

    while dets:
        best = dets.pop(0)
        final.append(best)

        restantes = []
        for d in dets:
            if d[4] != best[4]:
                restantes.append(d)
                continue
            if iou(best[:4], d[:4]) < iou_thresh:
                restantes.append(d)
        dets = restantes

    return final


def get_yolo_detections(frame, interpreter_data):
    """
    Devuelve lista de detecciones:
    [x_min, y_min, x_max, y_max, class_name, conf]
    usando el modelo YOLO TFLite.
    """
    interpreter, input_details, output_details, INPUT_SHAPE = interpreter_data

    # --- Preprocesamiento ---
    img = cv2.resize(frame, tuple(INPUT_SHAPE))
    inp = img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # [1, H, W, 3]

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    raw = interpreter.get_tensor(output_details[0]["index"])[0]
    print(f"[DEBUG] raw YOLO output shape: {raw.shape}")  # ej: (14, 8400)

    nc = len(config.YOLO_CLASSES)

    # Esperamos formato (4 + nc, N) = (14, 8400)
    if raw.ndim == 2 and raw.shape[0] == 4 + nc:
        raw = raw.T  # (N, 4+nc)
        print("[DEBUG] Interpretando salida como (C, N) -> (N, C)")
    elif raw.ndim == 2 and raw.shape[1] == 4 + nc:
        print("[DEBUG] Interpretando salida como (N, C)")
    else:
        raw = raw.reshape(-1, raw.shape[-1])
        print("[DEBUG] Reajustando salida a 2D:", raw.shape)

    pred = raw  # (N, 4+nc)

    H, W, _ = frame.shape
    frame_area = float(W * H)
    min_area = 0.03 * frame_area   # solo cajas > 3% del área

    candidates = []
    max_confs = []

    for det in pred:
        if det.shape[0] < 4 + nc:
            continue

        x, y, w, h = det[:4]
        class_scores_raw = det[4:]

        class_scores = sigmoid(class_scores_raw)
        conf = float(np.max(class_scores))
        cls_id = int(np.argmax(class_scores))
        max_confs.append(conf)

        if conf < config.CONFIDENCE_THRESHOLD:
            continue

        if cls_id < 0 or cls_id >= nc:
            continue

        class_name = config.YOLO_CLASSES[cls_id]

        x_min = int((x - w / 2) * W)
        y_min = int((y - h / 2) * H)
        x_max = int((x + w / 2) * W)
        y_max = int((y + h / 2) * H)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(W, x_max), min(H, y_max)

        if x_max <= x_min or y_max <= y_min:
            continue

        area = float((x_max - x_min) * (y_max - y_min))
        if area < min_area:
            continue

        candidates.append([x_min, y_min, x_max, y_max, class_name, conf])

    print(f"[DEBUG] Candidatos antes de NMS: {len(candidates)}")

    nms_dets = nms(candidates, iou_thresh=0.4)

    by_class = {}
    for det in sorted(nms_dets, key=lambda d: d[5], reverse=True):
        cls = det[4]
        by_class.setdefault(cls, [])
        if len(by_class[cls]) < 2:   # max 2 por clase
            by_class[cls].append(det)

    results = []
    for cls, dets in by_class.items():
        results.extend(dets)

    results = sorted(results, key=lambda d: d[5], reverse=True)[:10]

    if max_confs:
        print(f"[DEBUG] Max conf en este frame: {max(max_confs):.3f}")
    print(f"[DEBUG] Detecciones YOLO en este frame (finales): {len(results)}")

    return results


# ---------------------- MiDaS / Profundidad ---------------------- #

def get_depth_map(frame, midas_data):
    """Devuelve mapa de profundidad normalizado (uint8) o None."""
    if not config.USE_DEPTH:
        return None

    midas, transform, device = midas_data
    if midas is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)

    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = pred.cpu().numpy()
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def calculate_distance(box, depth_map):
    """
    Calcula una pseudo-distancia a partir del mapa de profundidad.
    Distancia aproximada, NO métrica exacta (pero sirve para saber si está cerca/lejos).
    """
    if depth_map is None:
        return None

    x1, y1, x2, y2 = box
    roi = depth_map[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    med = float(np.median(roi))
    if med <= 0:
        return None

    dist = 1.0 / (med * config.CALIBRATION_FACTOR)
    return dist