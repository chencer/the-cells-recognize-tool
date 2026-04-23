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
img = cv2.imread("test_cell.jpg", cv2.IMREAD_COLOR)
assert img is not None, "Image load failed"
print(f"Image: {img.shape[1]}x{img.shape[0]}, dtype={img.dtype}")

# --- Load Cellpose model ---
from cellpose import models
_orig_torch_load = torch.serialization.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
use_gpu = torch.cuda.is_available()
print(f"GPU: {use_gpu}")

model = models.CellposeModel(gpu=use_gpu, model_type="cyto3")
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
total_detected = len(cell_ids)
print(f"Total cells detected: {total_detected}")
assert total_detected > 0, "No cells detected"

# --- Convex hull completeness filter + brightness ranking ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cell_list = []
skipped_incomplete = 0

for cid in cell_ids:
    mask = (masks == cid).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        skipped_incomplete += 1
        continue
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        skipped_incomplete += 1
        continue
    mask_area = float(np.sum(mask > 0))
    completeness = mask_area / hull_area
    if completeness < 0.7:
        skipped_incomplete += 1
        continue

    cell_pixels = gray[mask > 0]
    if len(cell_pixels) > 50:
        sorted_pixels = np.sort(cell_pixels)[::-1]
        top_5_count = max(1, int(len(sorted_pixels) * 0.05))
        peak_brightness = np.mean(sorted_pixels[:top_5_count])
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cell_list.append({"brightness": peak_brightness, "pos": (cx, cy), "mask": mask})

cell_list.sort(key=lambda x: x["brightness"], reverse=True)

print(f"检测总数: {total_detected}  过滤掉(完整度<70%): {skipped_incomplete}  有效细胞: {len(cell_list)}")
assert len(cell_list) > 0, "No valid cells after filtering"

top = cell_list[0]
cx, cy = top["pos"]
brightness_val = int(top["brightness"])
print(f"最亮细胞坐标: ({cx}, {cy})  亮度值: {brightness_val}")

# --- Annotated output: only the brightest cell ---
res = img.copy()
contours, _ = cv2.findContours(top["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(res, contours, -1, (0, 0, 255), 2)
cv2.circle(res, (cx, cy), 5, (0, 0, 255), -1)
cv2.putText(res, f"({cx}, {cy})", (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("result.jpg", res)
print("Result saved: result.jpg")
print("ALL TESTS PASSED")
