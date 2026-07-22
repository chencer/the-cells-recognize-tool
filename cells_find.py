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


# --- 配置加载 ---
def load_settings():
    defaults = {
        'diameter': 120, 'flow_threshold': 0.95, 'cellprob_threshold': 1.0,
        'min_size': 200, 'resample': 0, 'niter': 200,
        'hull_comp': 0.85, 'circle_comp': 0.65, 'dark_threshold': 15,
        'area_ratio': 0.15, 'circularity': 0.5, 'min_pixels': 50,
        'brightness_top_pct': 0.05,
        'top_n': 3,
        'resize_area_threshold': 9000000, 'resize_max_side': 4096,
        'crop_pad': 2.0, 'contour_thickness': 2, 'font_scale': 0.7,
        'sort_descending': 1, 'enable_sort': 1, 'model_name': 'cyto3',
        'bad_dark_threshold': 60, 'bad_dark_ratio': 0.10,
        'bad_hull_threshold': 0.90, 'enable_bad_detection': 1,
        'bad_inner_ratio': 0.6,
        'hole_rel_ratio': 0.35, 'hole_min_area_ratio': 0.01,
        'hole_extent_min': 0.35, 'hole_area_ratio': 0.03,
        'defect_depth_ratio': 0.35,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.txt')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key, val = key.strip(), val.strip()
                    if key in defaults:
                        if key == 'model_name':
                            defaults[key] = val
                        elif isinstance(defaults[key], int):
                            defaults[key] = int(val)
                        elif isinstance(defaults[key], float):
                            defaults[key] = float(val)
        print(f"✅ 已读取配置: {path}", flush=True)
    else:
        print(f"⚠️ 未找到 settings.txt，使用默认参数", flush=True)
    return defaults


# --- 模型加载 ---
def load_model(settings):
    print("正在加载模型...", flush=True)
    ssl._create_default_https_context = ssl._create_unverified_context

    if torch.cuda.is_available():
        use_gpu = True
        print("  设备: CUDA GPU", flush=True)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_gpu = True
        print("  设备: Apple MPS (M系列)", flush=True)
    else:
        use_gpu = False
        print("  设备: CPU", flush=True)

    model_name = settings.get('model_name', 'cyto3')
    model_path = get_resource_path(model_name)
    if os.path.exists(model_path):
        m = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
        print(f"✅ 成功加载本地模型: {model_path}", flush=True)
    else:
        m = cp_models.CellposeModel(gpu=use_gpu, model_type=model_name)
        print(f"✅ 使用内置模型: {model_name}", flush=True)
    return m


# --- 坏细胞检测：内部空洞 + 边缘破损（两路径共用）---
def _detect_defects(local_gray, local_mask, er, settings):
    """返回 (is_bad, hole_ratio, defect_ratio, reason)"""
    m = local_mask.astype(np.uint8)
    mb = m > 0
    cell_area = int(mb.sum())
    if cell_area == 0:
        return False, 0.0, 0.0, ""
    reason = []

    # 线A：内部实心黑洞（相对亮度 + 连通域面积 + 实心度）
    core_ref = float(np.median(local_gray[mb]))
    dark_thr = core_ref * settings['hole_rel_ratio']
    dark_bin = ((local_gray < dark_thr) & mb).astype(np.uint8)
    hole_area = 0
    if dark_bin.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark_bin, 8)
        min_a = cell_area * settings['hole_min_area_ratio']
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_a:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            extent = a / (bw * bh) if bw * bh > 0 else 0.0
            if extent >= settings['hole_extent_min']:
                hole_area += a
    hole_ratio = hole_area / cell_area
    if hole_ratio > settings['hole_area_ratio']:
        reason.append('HOLE')

    # 线B：边缘缺口 / 形状破碎（凸包缺陷最大深度 / er）
    defect_ratio = 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) >= 4:
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3:
                try:
                    d = cv2.convexityDefects(cnt, hull)
                    if d is not None and er > 0:
                        defect_ratio = float(d[:, 0, 3].max()) / 256.0 / er
                except cv2.error:
                    pass
    if defect_ratio > settings['defect_depth_ratio']:
        reason.append('BROKEN')

    return (len(reason) > 0), hole_ratio, defect_ratio, '+'.join(reason)


# --- 过滤链（基于 cellpose mask）---
def _filter_and_rank_mask(masks, raw_image, settings):
    gray           = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    H_img, W_img   = gray.shape
    cell_ids       = np.unique(masks)[1:]
    total_detected = len(cell_ids)

    hull_comp_thr   = settings['hull_comp']
    circle_comp_thr = settings['circle_comp']
    area_ratio      = settings['area_ratio']
    circ_thr        = settings['circularity']
    min_pixels      = settings['min_pixels']
    top_pct         = settings['brightness_top_pct']

    # Step1: 完整度过滤
    candidates = []
    for cid in cell_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        mask_area   = float(np.sum(mask > 0))
        hull        = cv2.convexHull(contours[0])
        hull_area   = cv2.contourArea(hull)
        hull_comp   = mask_area / hull_area if hull_area > 0 else 0.0
        (_, _), er  = cv2.minEnclosingCircle(contours[0])
        circle_area = math.pi * er * er
        circle_comp = mask_area / circle_area if circle_area > 0 else 0.0
        if hull_comp < hull_comp_thr or circle_comp < circle_comp_thr:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        candidates.append({
            "mask": mask, "mask_area": mask_area,
            "contours": contours, "perimeter": perimeter, "er": er,
            "hull_comp": hull_comp,
        })
    print(f"  [Step1] 完整度过滤: {len(candidates)} / {total_detected}", flush=True)

    # Step2: 面积过滤
    if candidates:
        median_area = float(np.median([c["mask_area"] for c in candidates]))
        candidates  = [c for c in candidates if c["mask_area"] >= median_area * area_ratio]
    print(f"  [Step2] 面积过滤: {len(candidates)}", flush=True)

    # Step3: 圆形度过滤
    filtered = []
    for c in candidates:
        circ = (4 * math.pi * c["mask_area"] / (c["perimeter"] ** 2)
                if c["perimeter"] > 0 else 0.0)
        if circ >= circ_thr:
            filtered.append(c)
    candidates = filtered
    print(f"  [Step3] 圆形度过滤: {len(candidates)}", flush=True)

    # 亮度计算
    cell_list = []
    for c in candidates:
        M = cv2.moments(c["mask"])
        if M["m00"] <= 0:
            continue
        cx  = int(M["m10"] / M["m00"])
        cy  = int(M["m01"] / M["m00"])
        ir  = max(1, int(c["er"] * 0.8))
        rx1 = max(0, cx - ir);  ry1 = max(0, cy - ir)
        rx2 = min(W_img, cx + ir + 1);  ry2 = min(H_img, cy + ir + 1)
        roi_h, roi_w = ry2 - ry1, rx2 - rx1
        inner_mask   = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(inner_mask, (cx - rx1, cy - ry1), ir, 1, -1)
        sample_mask  = (inner_mask > 0) & (c["mask"][ry1:ry2, rx1:rx2] > 0)
        cell_pixels  = gray[ry1:ry2, rx1:rx2][sample_mask]
        if len(cell_pixels) > min_pixels:
            k    = max(1, int(len(cell_pixels) * top_pct))
            peak = float(np.mean(np.partition(cell_pixels, -k)[-k:]))
            if settings['enable_bad_detection']:
                ys0, xs0 = np.where(c["mask"] > 0)
                ly0, lx0 = int(ys0.min()), int(xs0.min())
                ly1, lx1 = int(ys0.max()), int(xs0.max())
                local_mask = c["mask"][ly0:ly1 + 1, lx0:lx1 + 1]
                local_gray = gray[ly0:ly1 + 1, lx0:lx1 + 1]
                is_bad, hole_ratio, defect_ratio, bad_reason = _detect_defects(
                    local_gray, local_mask, c["er"], settings)
            else:
                is_bad, hole_ratio, defect_ratio, bad_reason = False, 0.0, 0.0, ""
            cell_list.append({
                "brightness": peak, "pos": (cx, cy),
                "contours": c["contours"], "mask": c["mask"], "er": c["er"],
                "is_bad": is_bad, "dark_ratio": hole_ratio,
                "defect_ratio": defect_ratio, "reason": bad_reason,
            })

    if settings['enable_sort']:
        cell_list.sort(key=lambda x: x["brightness"], reverse=bool(settings['sort_descending']))
    print(f"  [Step4] 有效细胞: {len(cell_list)}", flush=True)
    return cell_list


# --- 单张图片处理 ---
def process_image(model, image_path, results_dir, settings):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n[{stem}] 处理中...", flush=True)

    raw_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw_image is None:
        print("  ❌ 无法读取图片，跳过", flush=True)
        return

    orig_h, orig_w = raw_image.shape[:2]
    scale = 1.0

    if orig_w * orig_h > settings['resize_area_threshold']:
        max_side = settings['resize_max_side']
        scale = min(1.0, max_side / max(orig_w, orig_h))
        if scale < 1.0:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  大图缩放: {orig_w}x{orig_h} → {new_w}x{new_h} (scale={scale:.4f})", flush=True)
        else:
            work_image = raw_image
    else:
        work_image = raw_image

    H_img, W_img = work_image.shape[:2]
    top_n = settings['top_n']

    save_dir = os.path.join(results_dir, stem)
    os.makedirs(save_dir, exist_ok=True)

    masks = model.eval(
        work_image,
        diameter=settings['diameter'],
        flow_threshold=settings['flow_threshold'],
        cellprob_threshold=settings['cellprob_threshold'],
        min_size=settings['min_size'],
        niter=settings['niter'],
        resample=bool(settings['resample']),
    )[0]
    if masks.shape != (H_img, W_img):
        masks = cv2.resize(masks, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    cell_list = _filter_and_rank_mask(masks, work_image, settings)
    total_detected = len(np.unique(masks)) - 1

    thickness  = settings['contour_thickness']
    font_scale = settings['font_scale']

    # ── 标注结果图 ────────────────────────────────────────────────────────────
    res_img = work_image.copy()
    if cell_list:
        bad_cells    = [c for c in cell_list if c.get('is_bad', False)]
        normal_cells = [c for c in cell_list if not c.get('is_bad', False)]
        top_cells    = normal_cells[:top_n]
        other_normal = normal_cells[top_n:]

        # 暗区压暗遮罩
        for cell in cell_list:
            cx, cy  = cell["pos"]
            er_int  = max(1, int(cell["er"]))
            ir_int  = max(1, int(cell["er"] * 0.8))
            outer_m = np.zeros((H_img, W_img), dtype=np.uint8)
            inner_m = np.zeros((H_img, W_img), dtype=np.uint8)
            cv2.circle(outer_m, (cx, cy), er_int, 1, -1)
            cv2.circle(inner_m, (cx, cy), ir_int, 1, -1)
            ring = (outer_m > 0) & (inner_m == 0) & (cell["mask"] > 0)
            res_img[ring] = (res_img[ring] * 0.6).astype(np.uint8)

        # 红色：坏细胞
        for cell in bad_cells:
            cx, cy    = cell["pos"]
            bad_label = cell.get('reason') or 'BAD'
            cv2.drawContours(res_img, cell["contours"], -1, (0, 0, 255), thickness)
            cv2.putText(res_img, bad_label, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 黄色：其余正常细胞
        for idx, cell in enumerate(other_normal, start=top_n + 1):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), thickness)
            cv2.putText(res_img, str(idx), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # 绿色：top_n 正常细胞
        for rank, cell in enumerate(top_cells, start=1):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 0), thickness)
            cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
            label = f"{rank} | {int(cell['brightness'])}"
            cv2.putText(res_img, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    print(f"  [Step5] 标注完成", flush=True)

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    print(f"  [Step6] 保存中...", flush=True)
    cv2.imencode('.png', res_img)[1].tofile(
        os.path.join(save_dir, f"{stem}_result.png"))

    crop_dir = os.path.join(save_dir, "crop")
    os.makedirs(crop_dir, exist_ok=True)
    _top_for_crop = [c for c in cell_list if not c.get('is_bad', False)][:top_n]
    for rank, cell in enumerate(_top_for_crop, start=1):
        cx, cy = cell["pos"]
        pad    = int(cell["er"] * settings['crop_pad'])
        y1     = max(0, cy - pad)
        y2     = min(H_img, cy + pad)
        x1     = max(0, cx - pad)
        x2     = min(W_img, cx + pad)
        crop   = res_img[y1:y2, x1:x2].copy()
        cv2.putText(crop, f"Top {rank}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imencode('.png', crop)[1].tofile(
            os.path.join(crop_dir, f"top{rank}.png"))
        print(f"  Top{rank} 裁剪图已保存", flush=True)

    csv_header = "编号,状态,标记原因,直径(px),坐标X,坐标Y,亮度值,空洞比例,缺口比例\n"
    with open(os.path.join(save_dir, f"{stem}_data.csv"), 'w', encoding='utf-8') as f:
        f.write(f"# 坐标系: 缩放后图片 ({W_img}x{H_img})\n")
        f.write(csv_header)
        for i, cell in enumerate(cell_list, start=1):
            cx, cy       = cell["pos"]
            diameter     = round(cell["er"] * 2, 1)
            status       = '!!!BAD!!!' if cell.get('is_bad', False) else 'OK'
            reason       = cell.get('reason', '') or '-'
            hole_ratio   = round(cell.get('dark_ratio', 0.0), 4)
            defect_ratio = round(cell.get('defect_ratio', 0.0), 4)
            f.write(f"{i},{status},{reason},{diameter},{cx},{cy},{int(cell['brightness'])},{hole_ratio},{defect_ratio}\n")

    with open(os.path.join(save_dir, f"{stem}_data_original.csv"), 'w', encoding='utf-8') as f:
        f.write(f"# 坐标系: 原图 ({orig_w}x{orig_h}), scale={scale:.4f}\n")
        f.write(csv_header)
        for i, cell in enumerate(cell_list, start=1):
            cx, cy       = cell["pos"]
            orig_cx      = int(cx / scale)
            orig_cy      = int(cy / scale)
            orig_diameter = round(cell["er"] * 2 / scale, 1)
            status       = '!!!BAD!!!' if cell.get('is_bad', False) else 'OK'
            reason       = cell.get('reason', '') or '-'
            hole_ratio   = round(cell.get('dark_ratio', 0.0), 4)
            defect_ratio = round(cell.get('defect_ratio', 0.0), 4)
            f.write(f"{i},{status},{reason},{orig_diameter},{orig_cx},{orig_cy},{int(cell['brightness'])},{hole_ratio},{defect_ratio}\n")

    top1    = cell_list[0] if cell_list else None
    summary = f"  检测: {total_detected}  有效: {len(cell_list)}"
    if top1:
        cx1, cy1 = top1["pos"]
        summary += f"  #1 ({cx1}, {cy1}) 亮度={int(top1['brightness'])}"
    print(summary, flush=True)
    print(f"  → 已保存到 results/{stem}/", flush=True)


# --- 主流程 ---
def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    input_dir   = os.path.join(base_dir, "input")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(input_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    settings = load_settings()
    print(f"  diameter={settings['diameter']}  top_n={settings['top_n']}  resize_threshold={settings['resize_area_threshold']}", flush=True)

    exts   = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    images = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not images:
        print("input/ 目录为空，请放入图片后重新运行。", flush=True)
        return

    print(f"发现 {len(images)} 张图片：", flush=True)
    for img in images:
        print(f"  {os.path.basename(img)}", flush=True)

    model   = load_model(settings)
    t_start = time.time()

    for img_path in images:
        process_image(model, img_path, results_dir, settings)

    elapsed = time.time() - t_start
    print(f"\n✅ 全部完成，共耗时 {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
