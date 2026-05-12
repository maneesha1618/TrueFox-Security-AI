# ============================================================
# scripts/fight_detector.py
# TrueFox AI — Intelligent Camera Security System
# MobileNetV3 inference — classifies fight vs nonfight
# ============================================================

import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FIGHT_CONF_THRESHOLD,
    FIGHT_IMG_SIZE,
    FIGHT_NUM_FRAMES,
    FIGHT_MEAN,
    FIGHT_STD,
    COLORS
)

# ── Device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform — must match training exactly ──────────────────
fight_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FIGHT_IMG_SIZE, FIGHT_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(FIGHT_MEAN, FIGHT_STD)
])

# ── Rolling frame buffer ─────────────────────────────────────
frame_buffer = deque(maxlen=FIGHT_NUM_FRAMES)


def preprocess_frame(frame):
    """
    Convert BGR OpenCV frame to normalized tensor.
    """
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = fight_transform(rgb)
    return tensor


def run_fight_detection(frame, fight_model):
    """
    Run fight classification on a single frame.
    Uses a rolling buffer — more stable than single frame.

    Args:
        frame       : BGR numpy array (OpenCV frame)
        fight_model : loaded MobileNetV3 model

    Returns:
        result : dict with keys:
                 {label, class_id, confidence, is_fight}
    """
    # Add frame to rolling buffer
    frame_buffer.append(preprocess_frame(frame))

    # Use middle frame of buffer as input
    mid_idx = len(frame_buffer) // 2
    input_tensor = frame_buffer[mid_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        output  = fight_model(input_tensor)
        probs   = torch.softmax(output, dim=1)[0]
        cls_id  = int(probs.argmax())
        conf    = float(probs[cls_id])

    is_fight = (cls_id == 1 and conf >= FIGHT_CONF_THRESHOLD)
    label    = "Fight" if cls_id == 1 else "NonFight"

    return {
        "class_id"   : cls_id,
        "label"      : label,
        "confidence" : round(conf, 3),
        "is_fight"   : is_fight,
        "fight_prob" : round(float(probs[1]), 3)  # always fight probability
    }


def draw_fight_status(frame, fight_result):
    """
    Draw fight classification status on top-right of frame.

    Args:
        frame        : BGR numpy array
        fight_result : dict from run_fight_detection()

    Returns:
        annotated frame
    """
    annotated = frame.copy()
    h, w      = annotated.shape[:2]

    label = fight_result["label"]
    conf  = fight_result["confidence"]
    prob  = fight_result["fight_prob"]

    # Choose color based on result
    if fight_result["is_fight"]:
        color = COLORS["fight"]
        text  = f"FIGHT {prob:.0%}"
    else:
        color = COLORS["nonfight"]
        text  = f"Normal {1-prob:.0%}"

    # Draw status box top-right
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x1 = w - tw - 20
    y1 = 10
    cv2.rectangle(annotated,
                  (x1 - 5, y1),
                  (x1 + tw + 5, y1 + th + 10),
                  color, -1)
    cv2.putText(annotated, text,
                (x1, y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLORS["text"], 2)

    return annotated