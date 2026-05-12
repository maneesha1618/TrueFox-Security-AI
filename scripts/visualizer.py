# ============================================================
# scripts/visualizer.py
# TrueFox AI — Intelligent Camera Security System
# Draws all annotations, alerts, and status overlays on frames
# ============================================================

import cv2
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLORS


def draw_weapon_boxes(frame, detections):
    """
    Draw bounding boxes + labels for weapon detections.

    Args:
        frame      : BGR numpy array
        detections : list of dicts from run_weapon_detection()

    Returns:
        annotated frame
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name  = det["class_name"]
        conf  = det["confidence"]
        color = COLORS.get(name, COLORS["box"])

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated,
                      (x1, y1 - th - 10),
                      (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(annotated, label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS["text"], 2)

    return annotated


def draw_fight_status(frame, fight_result):
    """
    Draw fight classification badge on top-right corner.

    Args:
        frame        : BGR numpy array
        fight_result : dict from run_fight_detection()

    Returns:
        annotated frame
    """
    annotated  = frame.copy()
    h, w       = annotated.shape[:2]
    prob       = fight_result["fight_prob"]
    is_fight   = fight_result["is_fight"]

    text  = f"FIGHT {prob:.0%}" if is_fight else f"Normal {1-prob:.0%}"
    color = COLORS["fight"] if is_fight else COLORS["nonfight"]

    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x1 = w - tw - 20
    y1 = 10
    cv2.rectangle(annotated,
                  (x1 - 6, y1),
                  (x1 + tw + 6, y1 + th + 12),
                  color, -1)
    cv2.putText(annotated, text,
                (x1, y1 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLORS["text"], 2)

    return annotated


def draw_alert_banner(frame, active_alerts):
    """
    Draw a red alert banner at the bottom of the frame
    when any alert is active.

    Args:
        frame         : BGR numpy array
        active_alerts : list of active alert type strings

    Returns:
        annotated frame
    """
    if not active_alerts:
        return frame

    annotated = frame.copy()
    h, w      = annotated.shape[:2]

    # Banner background
    banner_h = 45
    overlay  = annotated.copy()
    cv2.rectangle(overlay,
                  (0, h - banner_h),
                  (w, h),
                  COLORS["alert_bg"], -1)
    cv2.addWeighted(overlay, 0.75,
                    annotated, 0.25, 0, annotated)

    # Alert text
    alert_text = " | ".join([
        f"!! {a.upper()} DETECTED"
        for a in active_alerts
    ])
    (tw, th), _ = cv2.getTextSize(
        alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    tx = (w - tw) // 2
    ty = h - banner_h + th + 10
    cv2.putText(annotated, alert_text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, COLORS["text"], 2)

    return annotated


def draw_frame_info(frame, frame_id, fps):
    """
    Draw frame number + FPS counter on top-left.

    Args:
        frame    : BGR numpy array
        frame_id : current frame number
        fps      : current inference FPS

    Returns:
        annotated frame
    """
    annotated = frame.copy()

    info_lines = [
        f"Frame : {frame_id}",
        f"FPS   : {fps:.1f}",
    ]

    y = 25
    for line in info_lines:
        cv2.putText(annotated, line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
        y += 22

    return annotated


def draw_system_label(frame):
    """
    Draw system name watermark on top-center.
    """
    annotated = frame.copy()
    h, w      = annotated.shape[:2]
    label     = "TrueFox AI — Security Monitor"

    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = (w - tw) // 2
    cv2.putText(annotated, label,
                (tx, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 180, 180), 1)

    return annotated


def annotate_frame(frame, detections,
                   fight_result, active_alerts,
                   frame_id, fps):
    """
    Master annotation function — applies all overlays.

    Call this once per frame with all detection results.

    Args:
        frame         : raw BGR frame
        detections    : weapon detections list
        fight_result  : fight detection dict
        active_alerts : list of active alert strings
        frame_id      : frame number
        fps           : current FPS

    Returns:
        fully annotated frame
    """
    frame = draw_weapon_boxes(frame, detections)
    frame = draw_fight_status(frame, fight_result)
    frame = draw_alert_banner(frame, active_alerts)
    frame = draw_frame_info(frame, frame_id, fps)
    frame = draw_system_label(frame)
    return frame