# ============================================================
# config.py
# TrueFox AI — Intelligent Camera Security System
# All paths, thresholds, and constants in one place
# ============================================================

import os

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data_sample")

# ── Model Paths ──────────────────────────────────────────────
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "weapon_yolov8m_best.pt")
FIGHT_MODEL_PATH  = os.path.join(MODELS_DIR, "fight_best.pth")

# ── Model Input Settings ─────────────────────────────────────
YOLO_IMG_SIZE   = 640       # YOLOv8 input resolution
FIGHT_IMG_SIZE  = 224       # MobileNetV3 input resolution
FIGHT_NUM_FRAMES = 16       # frames in sliding window

# ── Detection Thresholds ─────────────────────────────────────
WEAPON_CONF_THRESHOLD = 0.45   # min confidence to count as weapon
FIGHT_CONF_THRESHOLD  = 0.65   # min probability to count as fight

# consecutive frames a condition must hold before alert fires
ALERT_CONSECUTIVE_FRAMES = 3

# ── Class Names ──────────────────────────────────────────────
WEAPON_CLASSES = {
    0: "pistol",
    1: "knife"
}

FIGHT_CLASSES = {
    0: "NonFight",
    1: "Fight"
}

# ── Alert Messages ───────────────────────────────────────────
ALERT_MESSAGES = {
    "pistol"  : "ALERT: Armed person detected (pistol)",
    "knife"   : "ALERT: Armed person detected (knife)",
    "fight"   : "ALERT: Fight/violence detected",
}

# ── Colors (BGR for OpenCV) ──────────────────────────────────
COLORS = {
    "pistol"    : (0, 0, 255),      # red
    "knife"     : (0, 140, 255),    # orange
    "fight"     : (0, 0, 200),      # dark red
    "nonfight"  : (0, 200, 0),      # green
    "alert_bg"  : (0, 0, 180),      # alert banner background
    "text"      : (255, 255, 255),  # white text
    "box"       : (255, 255, 0),    # yellow box default
}

# ── Output Settings ──────────────────────────────────────────
OUTPUT_VIDEO_NAME = "output_annotated.avi"
OUTPUT_CSV_NAME   = "inference_log.csv"
OUTPUT_ALERT_DIR  = os.path.join(RESULTS_DIR, "alert_snapshots")

SAVE_ALERT_SNAPSHOTS = True     # save jpg when alert fires
SHOW_LIVE_PREVIEW    = True     # display cv2 window during inference

# ── Video Settings ───────────────────────────────────────────
TARGET_FPS    = 30              # output video FPS
FRAME_SKIP    = 1               # process every Nth frame (1 = all)

# ── Normalization for fight classifier ───────────────────────
FIGHT_MEAN = [0.485, 0.456, 0.406]
FIGHT_STD  = [0.229, 0.224, 0.225]

# ── Sanity check on startup ──────────────────────────────────
def verify_paths():
    errors = []
    if not os.path.exists(WEAPON_MODEL_PATH):
        errors.append(f"Weapon model not found: {WEAPON_MODEL_PATH}")
    if not os.path.exists(FIGHT_MODEL_PATH):
        errors.append(f"Fight model not found : {FIGHT_MODEL_PATH}")
    os.makedirs(RESULTS_DIR,    exist_ok=True)
    os.makedirs(OUTPUT_ALERT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,       exist_ok=True)
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return False
    print("[CONFIG] ✓ All paths verified")
    return True