import os
import cv2
import torch
import numpy as np
from cellpose import models as cp_models
import sys
import ssl
import math
import time


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# --- 模型加载 ---
def load_model():
    print("正在加载模型...")
    torch.serialization.add_safe_globals([cp_models.CellposeModel])
    import torch.serialization as _ts
    torch.load = lambda *a, **kw: _ts.load(*a, **kw, weights_only=False)
    ssl._create_default_https_context = ssl._create_unverified_context

    use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    model_path = get_resource_path("cyto3")
    if os.path.exists(model_path):
        m = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
        print(f"✅ 成功加载本地模型: {model_path}")
    else:
        m = cp_models.Cellpose(gpu=use_gpu, model_type="cyto3")
        print("✅ 使用系统默认路径加载 cyto3")
    return m


# --- 单张图片处理 ---
def process_image(model, image_path, results_dir):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n[{stem}] 处理中...")

    raw_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw_image is None:
        print(f"  ❌ 无法读取图片，跳过")
        return

    masks = model.eval(
        raw_image,
        diameter=120,
        channels=[0, 0],
        flow_threshold=0.95,
        cellprob_threshold=1.0,
        min_size=200,
        resample=True,
    )[0]

    res_img       = raw_image.copy()
    gray          = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    cell_ids      = np.unique(masks)[1:]
    total_detected = len(cell_ids)
    H_img, W_img  = gray.shape

    # ── Step 1: 完整度过滤 ────────────────────────────────────────────────────
    candidates = []
    for cid in cell_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        mask_area = float(np.sum(mask > 0))

        hull      = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        hull_comp = mask_area / hull_area if hull_area > 0 else 0.0

        (_, _), er  = cv2.minEnclosingCircle(contours[0])
        circle_area = math.pi * er * er
        circle_comp = mask_area / circle_area if circle_area > 0 else 0.0

        if hull_comp < 0.85 or circle_comp < 0.65:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        candidates.append({
            "cid": cid, "mask": mask, "mask_area": mask_area,
            "contours": contours, "perimeter": perimeter, "er": er,
        })

    # ── Step 2: 面积过滤 ──────────────────────────────────────────────────────
    if candidates:
        median_area = float(np.median([c["mask_area"] for c in candidates]))
        area_thresh = median_area * 0.15
        candidates  = [c for c in candidates if c["mask_area"] >= area_thresh]

    # ── Step 3: 圆形度过滤 ────────────────────────────────────────────────────
    filtered = []
    for c in candidates:
        circ = (4 * math.pi * c["mask_area"] / (c["perimeter"] ** 2)
                if c["perimeter"] > 0 else 0.0)
        if circ >= 0.5:
            filtered.append(c)
    candidates = filtered

    # ── 亮度计算 ──────────────────────────────────────────────────────────────
    cell_list = []
    for c in candidates:
        M = cv2.moments(c["mask"])
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        ir  = max(1, int(c["er"] * 0.8))
        rx1 = max(0, cx - ir);  ry1 = max(0, cy - ir)
        rx2 = min(W_img, cx + ir + 1);  ry2 = min(H_img, cy + ir + 1)
        roi_h, roi_w = ry2 - ry1, rx2 - rx1

        inner_mask  = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(inner_mask, (cx - rx1, cy - ry1), ir, 1, -1)
        sample_mask = (inner_mask > 0) & (c["mask"][ry1:ry2, rx1:rx2] > 0)
        cell_pixels = gray[ry1:ry2, rx1:rx2][sample_mask]

        if len(cell_pixels) > 50:
            k    = max(1, int(len(cell_pixels) * 0.05))
            peak = float(np.mean(np.partition(cell_pixels, -k)[-k:]))
            cell_list.append({
                "brightness": peak, "pos": (cx, cy),
                "contours": c["contours"], "mask": c["mask"], "er": c["er"],
            })

    cell_list.sort(key=lambda x: x["brightness"], reverse=True)

    # ── 标注结果图 ────────────────────────────────────────────────────────────
    if cell_list:
        for idx, cell in enumerate(cell_list[3:], start=4):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), 2)
            cv2.putText(res_img, str(idx), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        for cell in cell_list:
            cx, cy   = cell["pos"]
            er_int   = max(1, int(cell["er"]))
            ir_int   = max(1, int(cell["er"] * 0.8))
            outer_m  = np.zeros((H_img, W_img), dtype=np.uint8)
            inner_m  = np.zeros((H_img, W_img), dtype=np.uint8)
            cv2.circle(outer_m, (cx, cy), er_int, 1, -1)
            cv2.circle(inner_m, (cx, cy), ir_int, 1, -1)
            ring = (outer_m > 0) & (inner_m == 0) & (cell["mask"] > 0)
            res_img[ring] = (res_img[ring] * 0.6).astype(np.uint8)

        for rank, cell in enumerate(cell_list[:3], start=1):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 0), 2)
            cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
            label = f"{rank} | {int(cell['brightness'])}"
            cv2.putText(res_img, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    save_dir = os.path.join(results_dir, stem)
    os.makedirs(save_dir, exist_ok=True)

    cv2.imencode('.png', res_img)[1].tofile(
        os.path.join(save_dir, f"{stem}_result.png"))

    with open(os.path.join(save_dir, f"{stem}_top3.csv"), 'w', encoding='utf-8') as f:
        f.write("排名,坐标X,坐标Y,亮度值\n")
        for i, cell in enumerate(cell_list[:3], start=1):
            cx, cy = cell["pos"]
            f.write(f"{i},{cx},{cy},{int(cell['brightness'])}\n")

    with open(os.path.join(save_dir, f"{stem}_data.csv"), 'w', encoding='utf-8') as f:
        f.write("编号,直径(px),坐标X,坐标Y,亮度值\n")
        for i, cell in enumerate(cell_list, start=1):
            cx, cy   = cell["pos"]
            diameter = round(cell["er"] * 2, 1)
            f.write(f"{i},{diameter},{cx},{cy},{int(cell['brightness'])}\n")

    top1 = cell_list[0] if cell_list else None
    summary = f"  检测: {total_detected}  有效: {len(cell_list)}"
    if top1:
        cx1, cy1 = top1["pos"]
        summary += f"  #1 ({cx1}, {cy1}) 亮度={int(top1['brightness'])}"
    print(summary)
    print(f"  → 已保存到 results/{stem}/")


# --- 主流程 ---
def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    input_dir   = os.path.join(base_dir, "input")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(input_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    exts   = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    images = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not images:
        print("input/ 目录为空，请放入图片后重新运行。")
        return

    print(f"发现 {len(images)} 张图片：")
    for img in images:
        print(f"  {os.path.basename(img)}")

    ans = input("\n按 y 开始处理：").strip().lower()
    if ans != 'y':
        print("已取消。")
        return

    model   = load_model()
    t_start = time.time()

    for img_path in images:
        process_image(model, img_path, results_dir)

    elapsed = time.time() - t_start
    print(f"\n✅ 全部完成，共耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
