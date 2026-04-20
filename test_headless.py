import os
import sys
import ssl
import cv2
import numpy as np
import torch

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print(f"Torch:  {torch.__version__}")

# --- Load image ---
img = cv2.imread("test_cell.tif", cv2.IMREAD_COLOR)
assert img is not None, "Image load failed"
print(f"Image: {img.shape[1]}x{img.shape[0]}, dtype={img.dtype}")

# --- Load Cellpose model ---
from cellpose import models
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)
use_gpu = torch.cuda.is_available()
print(f"GPU: {use_gpu}")

model_path = "cyto3"
if os.path.exists(model_path):
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    print("Model: local cyto3")
else:
    model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
    print("Model: system cyto3")

# --- Segmentation ---
print("Running segmentation...")
masks, flows, styles = model.eval(
    img,
    diameter=120,
    channels=[0, 0],
    flow_threshold=0.95,
    cellprob_threshold=1.0,
    min_size=200,
    resample=True
)[:3]

cell_ids = np.unique(masks)[1:]
print(f"Total cells detected: {len(cell_ids)}")
assert len(cell_ids) > 0, "No cells detected"

# --- Edge filter + brightness ranking ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cell_list = []
skipped_edge = 0

for cid in cell_ids:
    mask = (masks == cid).astype(np.uint8)
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        skipped_edge += 1
        continue
    cell_pixels = gray[mask > 0]
    if len(cell_pixels) > 50:
        sorted_pixels = np.sort(cell_pixels)[::-1]
        top_10_count = max(1, int(len(sorted_pixels) * 0.1))
        core_brightness = np.mean(sorted_pixels[:top_10_count])
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cell_list.append({"brightness": core_brightness, "pos": (cx, cy)})

cell_list.sort(key=lambda x: x["brightness"], reverse=True)

print(f"Edge-truncated filtered: {skipped_edge}")
print(f"Valid cells: {len(cell_list)}")
assert len(cell_list) > 0, "No valid cells after filtering"

top = cell_list[0]
print(f"Brightest cell: pos={top['pos']}  brightness={int(top['brightness'])}")

# --- Annotated output ---
res = img.copy()
for i, cid in enumerate(np.unique(masks)[1:]):
    mask = (masks == cid).astype(np.uint8)
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        continue
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 0, 255) if i < 2 else (0, 255, 255)
    cv2.drawContours(res, contours, -1, color, 2)

cx, cy = top["pos"]
cv2.putText(res, f"BRIGHTEST ({cx},{cy}) val:{int(top['brightness'])}",
            (cx - 60, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite("result.jpg", res)
print("Result saved: result.jpg")
print("ALL TESTS PASSED")
