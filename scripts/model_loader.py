# ============================================================
# scripts/model_loader.py
# TrueFox AI — Intelligent Camera Security System
# Loads both models once at startup
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WEAPON_MODEL_PATH,
    FIGHT_MODEL_PATH,
    FIGHT_IMG_SIZE
)

# ── Device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Fight Classifier Architecture ────────────────────────────
# Must match exactly what was trained on Kaggle
class FightClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=None)   # no pretrained — we load our own weights
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def load_weapon_model():
    """
    Load fine-tuned YOLOv8m weapon detection model.
    Returns: YOLO model object
    """
    print(f"[MODEL] Loading weapon model from: {WEAPON_MODEL_PATH}")
    try:
        model = YOLO(WEAPON_MODEL_PATH)
        print(f"[MODEL] ✓ Weapon model loaded (YOLOv8m)")
        return model
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load weapon model: {e}")
        raise


def load_fight_model():
    """
    Load trained MobileNetV3 fight classifier.
    Returns: model in eval mode on correct device
    """
    print(f"[MODEL] Loading fight model from: {FIGHT_MODEL_PATH}")
    try:
        model = FightClassifier(num_classes=2, dropout=0.4)
        state_dict = torch.load(
            FIGHT_MODEL_PATH,
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"[MODEL] ✓ Fight model loaded (MobileNetV3-Small) on {device}")
        return model
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load fight model: {e}")
        raise


def load_all_models():
    """
    Load both models at once.
    Returns: (weapon_model, fight_model)
    """
    print("[MODEL] Loading all models...")
    print(f"[MODEL] Device: {device}")
    weapon_model = load_weapon_model()
    fight_model  = load_fight_model()
    print("[MODEL] ✓ All models ready\n")
    return weapon_model, fight_model