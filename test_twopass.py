import sys
import cv2
import numpy as np
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

print(f"Python: {sys.version}")

img = cv2.imread("test_cell.tif", cv2.IMREAD_COLOR)
assert img is not None, "Image load failed"
H, W = img.shape[:2]
print(f"Image: {W}x{H}")

from cellpose import models
from cellpose.utils import diameters as cp_diameters

_orig = torch.serialization.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})

use_gpu = torch.cuda.is_available()
model = models.CellposeModel(gpu=use_gpu, pretrained_model="cyto3")
print(f"Model loaded  GPU={use_gpu}")

# Pass 1
print("Pass 1: estimating diameter...")
masks_est, _, _ = model.eval(img, diameter=0, channels=[0, 0])
diam = float(np.median(cp_diameters(masks_est)))
if diam == 0 or np.isnan(diam):
    diam = 30.0
print(f"Cellpose 估算直径: {diam:.1f}px")

# correction
short_side = min(H, W)
scale = short_side / 1080.0
corrected_diam = max(20.0, min(diam * scale, 500.0))
print(f"校正直径: {corrected_diam:.1f}px  (short_side={short_side}, scale={scale:.2f})")

assert corrected_diam > 0 and not np.isnan(corrected_diam), "corrected_diam invalid"

# Pass 2
print("Pass 2: segmentation...")
masks = model.eval(
    img,
    diameter=corrected_diam,
    channels=[0, 0],
    flow_threshold=0.95,
    cellprob_threshold=1.0,
    min_size=int(corrected_diam ** 2 * 0.2),
    resample=True,
)[0]

cell_ids = np.unique(masks)[1:]
print(f"检测到细胞数: {len(cell_ids)}")
assert len(cell_ids) > 0, "No cells detected in Pass 2"

print("✅ 两遍识别测试通过")
