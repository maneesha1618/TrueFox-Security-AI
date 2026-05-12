# ============================================================
# scripts/weapon_detector.py
# TrueFox AI — Intelligent Camera Security System
# YOLOv8 inference — detects pistol and knife per frame
# ============================================================

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WEAPON_CONF_THRESHOLD,
    WEAPON_CLASSES,
    COLORS,
    YOLO_IMG_SIZE
)


def run_weapon_detection(frame, weapon_model):
    """
    Run YOLOv8 inference on a single frame.

    Args:
        frame        : BGR numpy array (OpenCV frame)
        weapon_model : loaded YOLO model

    Returns:
        detections : list of dicts, each with:
                     {class_id, class_name, confidence, bbox}
        annotated  : frame with boxes + labels drawn
    """
    results    = weapon_model(
        frame,
        imgsz   = YOLO_IMG_SIZE,
        conf    = WEAPON_CONF_THRESHOLD,
        verbose = False
    )[0]

    detections = []
    annotated  = frame.copy()

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        class_name = WEAPON_CLASSES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class_id"   : cls_id,
            "class_name" : class_name,
            "confidence" : round(conf, 3),
            "bbox"       : (x1, y1, x2, y2)
        })

        # Draw bounding box
        color = COLORS.get(class_name, COLORS["box"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label + confidence
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated,
                      (x1, y1 - th - 8),
                      (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(annotated, label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS["text"], 2)

    return detections, annotated


def get_weapon_summary(detections):
    """
    Summarize detections for alert engine.

    Returns:
        dict: {class_name: max_confidence}
        e.g. {"pistol": 0.87, "knife": 0.73}
    """
    summary = {}
    for det in detections:
        name = det["class_name"]
        conf = det["confidence"]
        if name not in summary or conf > summary[name]:
            summary[name] = conf
    return summary