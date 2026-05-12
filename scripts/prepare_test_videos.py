# ============================================================
# scripts/prepare_test_videos.py
# Converts downloaded clips/images into test videos
# ============================================================

import cv2
import os
import glob

DATA_DIR = "data_sample"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Convert fight AVI clips → keep as-is, just verify ───────
fight_clips = glob.glob(f"{DATA_DIR}/fight_*.avi")
print(f"✓ Fight clips found    : {len(fight_clips)}")
for clip in fight_clips:
    cap = cv2.VideoCapture(clip)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  {os.path.basename(clip)} — {frames} frames @ {fps:.0f}fps")

# ── Convert weapon images → short video ─────────────────────
weapon_imgs = glob.glob(f"{DATA_DIR}/weapon_*.jpg")
print(f"\n✓ Weapon images found  : {len(weapon_imgs)}")

if weapon_imgs:
    sample = cv2.imread(weapon_imgs[0])
    h, w   = sample.shape[:2]

    out = cv2.VideoWriter(
        f"{DATA_DIR}/weapon_test_video.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        5, (w, h)
    )
    for img_path in weapon_imgs:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (w, h))
            for _ in range(10):   # show each image for 10 frames
                out.write(img)
    out.release()
    print(f"✓ Weapon test video created: {DATA_DIR}/weapon_test_video.avi")

# ── Summary ──────────────────────────────────────────────────
print(f"\n── Files in data_sample/ ───────────────────")
for f in os.listdir(DATA_DIR):
    size = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f"  {f:40s} {size/1024:.1f} KB")