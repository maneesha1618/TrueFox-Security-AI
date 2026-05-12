# ============================================================
# scripts/logger.py
# TrueFox AI — Intelligent Camera Security System
# Saves CSV inference log + alert snapshot images
# ============================================================

import cv2
import csv
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RESULTS_DIR,
    OUTPUT_CSV_NAME,
    OUTPUT_ALERT_DIR,
    SAVE_ALERT_SNAPSHOTS
)


class InferenceLogger:
    """
    Handles all logging — CSV per frame + snapshot on alert.
    """

    def __init__(self, source_name="unknown"):
        self.source_name  = source_name
        self.csv_path     = os.path.join(RESULTS_DIR, OUTPUT_CSV_NAME)
        self.alert_dir    = OUTPUT_ALERT_DIR
        self.frame_count  = 0
        self.alert_count  = 0

        os.makedirs(self.alert_dir, exist_ok=True)

        # ── Initialize CSV ───────────────────────────────────
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow([
            "frame_id",
            "timestamp",
            "weapon_detected",
            "weapon_classes",
            "weapon_confidences",
            "fight_detected",
            "fight_probability",
            "alerts_fired",
            "source"
        ])
        print(f"[LOGGER] ✓ CSV log initialized: {self.csv_path}")

    def log_frame(self, frame_id, detections,
                  fight_result, fired_alerts):
        """
        Write one row to CSV for current frame.

        Args:
            frame_id      : int
            detections    : list from run_weapon_detection()
            fight_result  : dict from run_fight_detection()
            fired_alerts  : list from alert_engine.update()
        """
        weapon_detected = len(detections) > 0
        weapon_classes  = "|".join(
            [d["class_name"] for d in detections])
        weapon_confs    = "|".join(
            [str(d["confidence"]) for d in detections])
        fight_detected  = fight_result["is_fight"]
        fight_prob      = fight_result["fight_prob"]
        alerts_fired    = "|".join(
            [a["type"] for a in fired_alerts]) if fired_alerts else ""

        self._writer.writerow([
            frame_id,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            weapon_detected,
            weapon_classes,
            weapon_confs,
            fight_detected,
            fight_prob,
            alerts_fired,
            self.source_name
        ])
        self.frame_count += 1

        # Flush every 30 frames so data isn't lost on crash
        if self.frame_count % 30 == 0:
            self._csv_file.flush()

    def save_alert_snapshot(self, frame, alert_type, frame_id):
        """
        Save annotated frame as JPEG when alert fires.

        Args:
            frame      : annotated BGR numpy array
            alert_type : "pistol" / "knife" / "fight"
            frame_id   : current frame number
        """
        if not SAVE_ALERT_SNAPSHOTS:
            return

        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        filename   = f"alert_{alert_type}_{timestamp}_f{frame_id}.jpg"
        save_path  = os.path.join(self.alert_dir, filename)
        cv2.imwrite(save_path, frame)
        self.alert_count += 1
        print(f"[LOGGER] ✓ Snapshot saved: {filename}")

    def close(self):
        """
        Close CSV file and print summary.
        """
        self._csv_file.flush()
        self._csv_file.close()
        print(f"\n[LOGGER] ── Session Summary ──────────────────")
        print(f"[LOGGER]   Frames processed : {self.frame_count}")
        print(f"[LOGGER]   Alerts fired     : {self.alert_count}")
        print(f"[LOGGER]   CSV saved to     : {self.csv_path}")
        print(f"[LOGGER]   Snapshots saved  : {self.alert_dir}")