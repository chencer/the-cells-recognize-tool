import cv2
import numpy as np
import sys
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print(f"Torch: {torch.__version__}")

# --- 加载图片 ---
img = cv2.imread('test_cell.tif', cv2.IMREAD_COLOR)
assert img is not None, "图片加载失败"
print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}, dtype: {img.dtype}")

# --- 加载 Cellpose 模型 ---
from cellpose import models
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)
use_gpu = torch.cuda.is_available()
print(f"GPU 加速: {use_gpu}")

import os
model_path = "cyto3"
if os.path.exists(model_path):
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    print("模型: 本地 cyto3")
else:
    model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
    print("模型: 系统 cyto3")

# --- 细胞分割 ---
print("正在分割...")
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
print(f"分割到细胞总数: {len(cell_ids)}")
assert len(cell_ids) > 0, "未检测到任何细胞"

# --- 边缘过滤 + 亮度排名 ---
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

print(f"边缘截断过滤: {skipped_edge} 个")
print(f"有效细胞数: {len(cell_list)}")
assert len(cell_list) > 0, "过滤后无有效细胞"

top = cell_list[0]
print(f"\n最亮细胞: 坐标={top['pos']}  亮度={int(top['brightness'])}")

# --- 生成标注图 ---
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
cv2.imwrite('result.jpg', res)
print("标注图已保存: result.jpg")
print("\n✅ 全部测试通过")
