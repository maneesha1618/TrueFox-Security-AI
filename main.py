# Cell 1 — check all imports work
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sklearn
import pandas as pd

print("✓ All imports successful")
print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)

# Cell 2 — run YOLOv8 on a test image (downloads weights automatically)
model = YOLO("yolov8m.pt")  # ~50MB, auto-downloads on first run
results = model("https://ultralytics.com/images/zidane.jpg", verbose=False)

img = results[0].plot()
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("YOLOv8m baseline — sanity check passed")
plt.show()

# Cell 3 — confirm no GPU needed (CPU inference)
print("Device that will be used:", "GPU" if torch.cuda.is_available() else "CPU")
print("This is expected — training happens on Kaggle")