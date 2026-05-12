# ============================================================
# scripts/alert_engine.py
# TrueFox AI — Intelligent Camera Security System
# Combines both model outputs, manages alert state,
# deduplicates alerts, prints alert messages
# ============================================================

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALERT_CONSECUTIVE_FRAMES,
    ALERT_MESSAGES,
    WEAPON_CONF_THRESHOLD,
    FIGHT_CONF_THRESHOLD
)


class AlertEngine:
    """
    Manages alert state across frames.
    Fires alert only after N consecutive frames confirm detection.
    Prevents duplicate alerts from spamming every frame.
    """

    def __init__(self):
        # Consecutive frame counters
        self._counters = {
            "pistol" : 0,
            "knife"  : 0,
            "fight"  : 0,
        }
        # Track active alerts to avoid spam
        self._active_alerts = {
            "pistol" : False,
            "knife"  : False,
            "fight"  : False,
        }
        # Alert history log
        self.alert_log = []

        # Cooldown — seconds before same alert can fire again
        self._cooldown_secs  = 5.0
        self._last_alert_time = {
            "pistol" : 0.0,
            "knife"  : 0.0,
            "fight"  : 0.0,
        }

    def update(self, weapon_summary, fight_result, frame_id):
        """
        Update alert state with latest detections.

        Args:
            weapon_summary : dict from get_weapon_summary()
                             e.g. {"pistol": 0.87}
            fight_result   : dict from run_fight_detection()
            frame_id       : current frame number

        Returns:
            fired_alerts : list of alert dicts that fired this frame
        """
        fired_alerts = []
        now = time.time()

        # ── Weapon alerts ────────────────────────────────────
        for weapon in ["pistol", "knife"]:
            if weapon in weapon_summary and \
               weapon_summary[weapon] >= WEAPON_CONF_THRESHOLD:
                self._counters[weapon] += 1
            else:
                self._counters[weapon] = 0
                self._active_alerts[weapon] = False

            if self._counters[weapon] >= ALERT_CONSECUTIVE_FRAMES:
                cooldown_ok = (
                    now - self._last_alert_time[weapon]
                    >= self._cooldown_secs
                )
                if cooldown_ok:
                    conf    = weapon_summary[weapon]
                    message = (
                        f"{ALERT_MESSAGES[weapon]} "
                        f"with {conf:.0%} confidence"
                    )
                    alert = {
                        "frame_id"  : frame_id,
                        "type"      : weapon,
                        "confidence": conf,
                        "message"   : message,
                        "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    fired_alerts.append(alert)
                    self.alert_log.append(alert)
                    self._last_alert_time[weapon] = now
                    self._active_alerts[weapon]   = True
                    print(f"\n{'='*60}")
                    print(f"  {message}")
                    print(f"  Frame: {frame_id} | "
                          f"Time: {alert['timestamp']}")
                    print(f"{'='*60}\n")

        # ── Fight alert ──────────────────────────────────────
        if fight_result["is_fight"]:
            self._counters["fight"] += 1
        else:
            self._counters["fight"] = 0
            self._active_alerts["fight"] = False

        if self._counters["fight"] >= ALERT_CONSECUTIVE_FRAMES:
            cooldown_ok = (
                now - self._last_alert_time["fight"]
                >= self._cooldown_secs
            )
            if cooldown_ok:
                conf    = fight_result["fight_prob"]
                message = (
                    f"{ALERT_MESSAGES['fight']} "
                    f"with {conf:.0%} confidence"
                )
                alert = {
                    "frame_id"  : frame_id,
                    "type"      : "fight",
                    "confidence": conf,
                    "message"   : message,
                    "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S")
                }
                fired_alerts.append(alert)
                self.alert_log.append(alert)
                self._last_alert_time["fight"] = now
                self._active_alerts["fight"]   = True
                print(f"\n{'='*60}")
                print(f"  {message}")
                print(f"  Frame: {frame_id} | "
                      f"Time: {alert['timestamp']}")
                print(f"{'='*60}\n")

        return fired_alerts

    def get_active_alerts(self):
        """Return currently active alert types."""
        return [k for k, v in self._active_alerts.items() if v]

    def reset(self):
        """Reset all counters and active alerts."""
        for key in self._counters:
            self._counters[key]      = 0
            self._active_alerts[key] = False