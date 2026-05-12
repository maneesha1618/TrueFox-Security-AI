# ============================================================
# main.py
# TrueFox AI — Intelligent Camera Security System
# Entry point — wires all modules into a live inference loop
#
# Usage:
#   python main.py --source video.mp4
#   python main.py --source 0              # webcam
#   python main.py --source video.mp4 --save
#   python main.py --source video.mp4 --no-display
# ============================================================

import cv2
import time
import argparse
import os
import sys

from config import (
    verify_paths,
    RESULTS_DIR,
    OUTPUT_VIDEO_NAME,
    SHOW_LIVE_PREVIEW,
    TARGET_FPS,
    FRAME_SKIP
)
from scripts.model_loader    import load_all_models
from scripts.weapon_detector import run_weapon_detection, get_weapon_summary
from scripts.fight_detector  import run_fight_detection
from scripts.alert_engine    import AlertEngine
from scripts.visualizer      import annotate_frame
from scripts.logger          import InferenceLogger


# ── Argument parser ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueFox AI — Intelligent Camera Security System"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: path to video file or 0 for webcam"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output video to results/"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing live preview window"
    )
    parser.add_argument(
        "--conf-weapon", type=float, default=None,
        help="Override weapon confidence threshold (0.0–1.0)"
    )
    parser.add_argument(
        "--conf-fight", type=float, default=None,
        help="Override fight confidence threshold (0.0–1.0)"
    )
    return parser.parse_args()


# ── Video writer setup ───────────────────────────────────────
def setup_video_writer(cap, save):
    if not save:
        return None
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    out_path = os.path.join(RESULTS_DIR, OUTPUT_VIDEO_NAME)
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps, (w, h)
    )
    print(f"[MAIN] Saving output video to: {out_path}")
    return writer


# ── Main inference loop ──────────────────────────────────────
def run(args):
    # ── Verify all model files exist ─────────────────────────
    if not verify_paths():
        sys.exit(1)

    # ── Override thresholds if passed ────────────────────────
    if args.conf_weapon:
        import config
        config.WEAPON_CONF_THRESHOLD = args.conf_weapon
        print(f"[MAIN] Weapon threshold → {args.conf_weapon}")
    if args.conf_fight:
        import config
        config.FIGHT_CONF_THRESHOLD = args.conf_fight
        print(f"[MAIN] Fight threshold  → {args.conf_fight}")

    # ── Load both models ──────────────────────────────────────
    weapon_model, fight_model = load_all_models()

    # ── Open video source ─────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[MAIN ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_name     = os.path.basename(str(args.source)) \
                   if not str(args.source).isdigit() else "webcam"

    print(f"\n[MAIN] ── Starting inference ───────────────────")
    print(f"[MAIN]   Source  : {src_name}")
    print(f"[MAIN]   Frames  : {total_frames if total_frames > 0 else 'live'}")
    print(f"[MAIN]   Display : {not args.no_display}")
    print(f"[MAIN]   Save    : {args.save}")
    print(f"[MAIN]   Press Q to quit\n")

    # ── Initialize modules ────────────────────────────────────
    engine = AlertEngine()
    logger = InferenceLogger(source_name=src_name)
    writer = setup_video_writer(cap, args.save)

    # ── FPS tracking ──────────────────────────────────────────
    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.time()
    frame_id     = 0

    # ── Main loop ─────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[MAIN] End of video / no frame received.")
                break

            frame_id += 1

            # Skip frames if configured
            if frame_id % FRAME_SKIP != 0:
                continue

            # ── Run both models ───────────────────────────────
            detections, _ = run_weapon_detection(
                frame, weapon_model)
            fight_result  = run_fight_detection(
                frame, fight_model)

            # ── Alert engine ──────────────────────────────────
            summary      = get_weapon_summary(detections)
            fired_alerts = engine.update(
                summary, fight_result, frame_id)
            active_alerts = engine.get_active_alerts()

            # ── FPS calculation ───────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            # ── Annotate frame ────────────────────────────────
            annotated = annotate_frame(
                frame, detections, fight_result,
                active_alerts, frame_id, fps_display
            )

            # ── Log frame to CSV ──────────────────────────────
            logger.log_frame(
                frame_id, detections,
                fight_result, fired_alerts)

            # ── Save alert snapshots ──────────────────────────
            for alert in fired_alerts:
                logger.save_alert_snapshot(
                    annotated, alert["type"], frame_id)

            # ── Write to output video ─────────────────────────
            if writer:
                writer.write(annotated)

            # ── Show live preview ─────────────────────────────
            if not args.no_display:
                cv2.imshow("TrueFox AI — Security Monitor",
                           annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[MAIN] Quit signal received.")
                    break

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")

    finally:
        # ── Cleanup ───────────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.close()
        print("\n[MAIN] ✓ Inference complete.")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(args)