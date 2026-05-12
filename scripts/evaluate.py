# ============================================================
# scripts/evaluate.py
# TrueFox AI — Intelligent Camera Security System
# Generates full evaluation report from test runs
# Run: python scripts/evaluate.py
# ============================================================

import os
import sys
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, DATA_DIR
from scripts.model_loader    import load_all_models
from scripts.weapon_detector import run_weapon_detection, get_weapon_summary
from scripts.fight_detector  import run_fight_detection, frame_buffer
from scripts.alert_engine    import AlertEngine

# ── Ground truth labels for our test clips ───────────────────
# 1 = fight/weapon present, 0 = normal
TEST_CLIPS = [
    # (filename,         true_fight, true_weapon)
    ("fight_1.avi",      1,          0),
    ("fight_2.avi",      1,          0),
    ("fight_3.avi",      1,          0),
    ("nonfight_1.avi",   0,          0),
    ("nonfight_2.avi",   0,          0),
    ("nonfight_3.avi",   0,          0),
    ("weapon_test_video.avi", 0,     1),
]


def evaluate_clip(clip_path, weapon_model, fight_model):
    """
    Run inference on a clip and return per-frame results.
    Returns dict with fight predictions, weapon detections, timing.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[EVAL] Cannot open: {clip_path}")
        return None

    frame_buffer.clear()

    fight_preds   = []
    weapon_preds  = []
    frame_times   = []
    frame_id      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        t_start = time.time()

        # Run both models
        dets, _   = run_weapon_detection(frame, weapon_model)
        fight_res = run_fight_detection(frame, fight_model)

        t_end = time.time()
        frame_times.append(t_end - t_start)

        fight_preds.append(fight_res["fight_prob"])
        weapon_preds.append(1 if len(dets) > 0 else 0)

    cap.release()

    return {
        "fight_probs"  : fight_preds,
        "weapon_preds" : weapon_preds,
        "frame_times"  : frame_times,
        "num_frames"   : frame_id
    }


def clip_level_prediction(fight_probs, weapon_preds,
                           fight_thresh=0.65):
    """
    Aggregate frame-level predictions to clip-level.
    A clip is 'fight' if >30% of frames exceed threshold.
    A clip has 'weapon' if any frame detected one.
    """
    fight_frames = sum(1 for p in fight_probs
                       if p >= fight_thresh)
    fight_ratio  = fight_frames / len(fight_probs) \
                   if fight_probs else 0
    pred_fight   = 1 if fight_ratio > 0.30 else 0
    pred_weapon  = 1 if sum(weapon_preds) > 2 else 0
    return pred_fight, pred_weapon


def compute_metrics(y_true, y_pred, label=""):
    """Compute TP, FP, FN, TN, precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0

    return {
        "label"    : label,
        "TP"       : tp, "FP": fp,
        "FN"       : fn, "TN": tn,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "accuracy" : accuracy
    }


def plot_confusion_matrix(metrics, title, save_path):
    """Plot and save confusion matrix."""
    cm = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]]
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No", "Pred: Yes"])
    ax.set_yticklabels(["True: No", "True: Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2
                    else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved: {save_path}")


def plot_fps_chart(clip_names, fps_values, save_path):
    """Bar chart of FPS per clip."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2196F3" if "fight" in n else
               "#FF5722" if "weapon" in n else
               "#4CAF50" for n in clip_names]
    bars = ax.bar(range(len(clip_names)), fps_values,
                  color=colors)
    ax.set_xticks(range(len(clip_names)))
    ax.set_xticklabels(clip_names, rotation=30, ha="right")
    ax.set_ylabel("FPS")
    ax.set_title("Inference Speed per Clip")
    ax.axhline(y=np.mean(fps_values), color="red",
               linestyle="--", label=f"Avg: {np.mean(fps_values):.1f} FPS")
    ax.legend()
    # Labels on bars
    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", fontsize=9)
    patches = [
        mpatches.Patch(color="#2196F3", label="Fight clips"),
        mpatches.Patch(color="#4CAF50", label="NonFight clips"),
        mpatches.Patch(color="#FF5722", label="Weapon clips"),
    ]
    ax.legend(handles=patches)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved: {save_path}")


def run_evaluation():
    print("\n[EVAL] ── TrueFox AI — Evaluation Report ──────────")
    print("[EVAL] Loading models...")
    weapon_model, fight_model = load_all_models()

    fight_true,  fight_pred  = [], []
    weapon_true, weapon_pred = [], []
    clip_fps     = []
    clip_names   = []
    results_rows = []

    for clip_name, true_fight, true_weapon in TEST_CLIPS:
        clip_path = os.path.join(DATA_DIR, clip_name)
        if not os.path.exists(clip_path):
            print(f"[EVAL] Skipping (not found): {clip_name}")
            continue

        print(f"\n[EVAL] Processing: {clip_name}")
        result = evaluate_clip(clip_path, weapon_model, fight_model)
        if result is None:
            continue

        # Clip-level predictions
        pred_fight, pred_weapon = clip_level_prediction(
            result["fight_probs"],
            result["weapon_preds"]
        )

        # FPS
        avg_frame_time = np.mean(result["frame_times"])
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        fight_true.append(true_fight)
        fight_pred.append(pred_fight)
        weapon_true.append(true_weapon)
        weapon_pred.append(pred_weapon)
        clip_fps.append(fps)
        clip_names.append(clip_name.replace(".avi",""))

        results_rows.append({
            "clip"         : clip_name,
            "true_fight"   : true_fight,
            "pred_fight"   : pred_fight,
            "true_weapon"  : true_weapon,
            "pred_weapon"  : pred_weapon,
            "fps"          : round(fps, 2),
            "frames"       : result["num_frames"]
        })

        status_f = "✓" if pred_fight  == true_fight  else "✗"
        status_w = "✓" if pred_weapon == true_weapon else "✗"
        print(f"  Fight  : true={true_fight}  pred={pred_fight}  {status_f}")
        print(f"  Weapon : true={true_weapon} pred={pred_weapon} {status_w}")
        print(f"  FPS    : {fps:.2f}")

    # ── Compute metrics ──────────────────────────────────────
    fight_metrics  = compute_metrics(fight_true,  fight_pred,  "Fight")
    weapon_metrics = compute_metrics(weapon_true, weapon_pred, "Weapon")
    avg_fps        = np.mean(clip_fps) if clip_fps else 0

    # ── Plot confusion matrices ──────────────────────────────
    plot_confusion_matrix(
        fight_metrics,
        "Fight Detection — Confusion Matrix (clip level)",
        os.path.join(RESULTS_DIR, "eval_fight_confusion.png")
    )
    plot_confusion_matrix(
        weapon_metrics,
        "Weapon Detection — Confusion Matrix (clip level)",
        os.path.join(RESULTS_DIR, "eval_weapon_confusion.png")
    )

    # ── Plot FPS chart ────────────────────────────────────────
    if clip_fps:
        plot_fps_chart(
            clip_names, clip_fps,
            os.path.join(RESULTS_DIR, "eval_fps_chart.png")
        )

    # ── Write report ─────────────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("  TrueFox AI — Evaluation Report\n")
        f.write(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        f.write("\n-- System Info ------------------------------------------\n")
        f.write(f"  Weapon Model : YOLOv8m fine-tuned\n")
        f.write(f"  Fight Model  : MobileNetV3-Small\n")
        f.write(f"  Device       : CPU\n")
        f.write(f"  Test Clips   : {len(results_rows)}\n\n")

        f.write("\n-- Per Clip Results  ------------------------------------------\n")
        for row in results_rows:
            f.write(f"  {row['clip']:<30s} "
                    f"fight={'✓' if row['pred_fight']==row['true_fight'] else '✗'} "
                    f"weapon={'✓' if row['pred_weapon']==row['true_weapon'] else '✗'} "
                    f"fps={row['fps']:.1f}\n")

        f.write("\n-- Fight Detection Metrics  ------------------------------------------\n")
        f.write(f"  Precision : {fight_metrics['precision']:.3f}\n")
        f.write(f"  Recall    : {fight_metrics['recall']:.3f}\n")
        f.write(f"  F1 Score  : {fight_metrics['f1']:.3f}\n")
        f.write(f"  Accuracy  : {fight_metrics['accuracy']:.3f}\n")
        f.write(f"  TP={fight_metrics['TP']} "
                f"FP={fight_metrics['FP']} "
                f"FN={fight_metrics['FN']} "
                f"TN={fight_metrics['TN']}\n")

        f.write("\n-- Weapon Detection Metrics  ------------------------------------------\n")

        f.write(f"  Precision : {weapon_metrics['precision']:.3f}\n")
        f.write(f"  Recall    : {weapon_metrics['recall']:.3f}\n")
        f.write(f"  F1 Score  : {weapon_metrics['f1']:.3f}\n")
        f.write(f"  Accuracy  : {weapon_metrics['accuracy']:.3f}\n")
        f.write(f"  TP={weapon_metrics['TP']} "
                f"FP={weapon_metrics['FP']} "
                f"FN={weapon_metrics['FN']} "
                f"TN={weapon_metrics['TN']}\n")

        f.write("\n-- Inference Speed  ------------------------------------------\n")

        f.write(f"  Avg FPS       : {avg_fps:.2f}\n")
        f.write(f"  Min FPS       : {min(clip_fps):.2f}\n")
        f.write(f"  Max FPS       : {max(clip_fps):.2f}\n")
        f.write(f"  Avg ms/frame  : {1000/avg_fps:.1f}ms\n")

        f.write("\n-- Limitations  ------------------------------------------\n")
        f.write("  1. Fight classifier uses single frame (no temporal)\n")
        f.write("  2. CPU inference limits FPS vs GPU deployment\n")
        f.write("  3. Weapon model trained on still images, not CCTV\n")
        f.write("  4. Low light / occluded scenes reduce accuracy\n")

        f.write("\n-- Future Improvements  ------------------------------------------\n")

        f.write("  1. Use LSTM over frame sequences for fight detection\n")
        f.write("  2. Fine-tune on CCTV-specific footage\n")
        f.write("  3. Add person tracking with ByteTrack\n")
        f.write("  4. Deploy with TensorRT for real-time GPU inference\n")
        f.write("="*60 + "\n")

    print(f"\n[EVAL] ── Final Metrics ────────────────────────────")

    print(f"  Fight  → P={fight_metrics['precision']:.3f} "
          f"R={fight_metrics['recall']:.3f} "
          f"F1={fight_metrics['f1']:.3f}")
    print(f"  Weapon → P={weapon_metrics['precision']:.3f} "
          f"R={weapon_metrics['recall']:.3f} "
          f"F1={weapon_metrics['f1']:.3f}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"\n[EVAL] Report saved to: {report_path}")
    print("[EVAL] ✓ Evaluation complete")


if __name__ == "__main__":
    run_evaluation()