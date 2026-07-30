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
        'min_size': 200, 'niter': 200,
        'hull_comp': 0.85, 'circle_comp': 0.65, 'dark_threshold': 15,
        'area_ratio': 0.15, 'circularity': 0.5, 'min_pixels': 50,
        'brightness_top_pct': 0.05,
        'top_n': 3,
        'resize_scale': 0.15,
        'crop_pad': 2.0, 'contour_thickness': 2, 'font_scale': 0.7,
        'sort_descending': 1, 'enable_top_ranking': 1, 'enable_bad_ranking': 1,
        'model_name': 'cyto3',
        'bad_dark_threshold': 60, 'enable_bad_detection': 1,
        'bad_edge_margin': 20,
        'bad_blob_min_area': 0.01, 'bad_blob_compactness': 0.35,
        'bad_broken_area': 0.15,
        'bad_blob_angle_span': 180,
        'bad_blob_center_radius': 0.5,
        'bad_hole_area_center': 0.005, 'bad_hole_area_edge': 0.025,
        'bad_adaptive_threshold': 1, 'bad_adaptive_ratio': 0.5,
        'bad_adaptive_inner': 0.6, 'bad_adaptive_percentile': 75,
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


# --- 坏细胞检测：暗区连通域检测（两路径共用）---
def _detect_defects(gy, gx, gray, er, bbox, H_img, W_img, settings):
    """暗区连通域检测：区分环状暗环和团块状虫蛀/破损，按位置分级判定"""
    reasons = []
    cell_area = len(gy)
    max_blob_ratio = 0.0
    max_blob_angle_span = 0
    max_blob_pos = 0.0
    blob_count = 0
    local_cy = float(np.mean(gy - bbox[0]))
    local_cx = float(np.mean(gx - bbox[1]))

    ly0, lx0 = bbox[0], bbox[1]
    lh = bbox[2] - ly0 + 1
    lw = bbox[3] - lx0 + 1
    local_cell = np.zeros((lh, lw), dtype=np.uint8)
    local_cell[gy - ly0, gx - lx0] = 1
    local_gray = np.zeros((lh, lw), dtype=np.uint8)
    local_gray[gy - ly0, gx - lx0] = gray[gy, gx]

    # 计算暗块判定阈值
    if settings['bad_adaptive_threshold']:
        cy_c = float(np.mean(gy))
        cx_c = float(np.mean(gx))
        dist_all = np.sqrt((gy - cy_c) ** 2 + (gx - cx_c) ** 2)
        eff_r_full = np.sqrt(cell_area / np.pi)
        inner_sel = dist_all <= eff_r_full * settings['bad_adaptive_inner']
        if np.sum(inner_sel) > 10:
            inner_vals = gray[gy[inner_sel], gx[inner_sel]]
            # 用分位数而非平均值，避免暗斑拉低基准造成自我抵消
            base_brightness = float(np.percentile(
                inner_vals, settings['bad_adaptive_percentile']))
            dark_thr = base_brightness * settings['bad_adaptive_ratio']
        else:
            dark_thr = float(settings['bad_dark_threshold'])
    else:
        dark_thr = float(settings['bad_dark_threshold'])

    dark_mask = ((local_gray < dark_thr) & (local_cell > 0)).astype(np.uint8)

    if np.sum(dark_mask) > 0:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        min_area = cell_area * settings['bad_blob_min_area']
        for i in range(1, n_labels):
            blob_area = int(stats[i, cv2.CC_STAT_AREA])
            if blob_area < min_area:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            rect_area = bw * bh
            compactness = blob_area / rect_area if rect_area > 0 else 0.0
            blob_angle_span = 0
            if compactness < settings['bad_blob_compactness']:
                blob_pixels = np.argwhere(labels == i)
                if len(blob_pixels) == 0:
                    continue
                angles = np.arctan2(
                    blob_pixels[:, 0] - local_cy,
                    blob_pixels[:, 1] - local_cx
                )
                angles_deg = np.degrees((angles + 2 * np.pi) % (2 * np.pi))
                hist, _ = np.histogram(angles_deg, bins=36, range=(0, 360))
                blob_angle_span = int(np.sum(hist > 0)) * 10
                if blob_angle_span >= settings['bad_blob_angle_span']:
                    continue

            # 暗块中心到细胞中心的相对距离（0=中心，1=边缘）
            blob_cy = float(centroids[i][1])
            blob_cx = float(centroids[i][0])
            blob_dist = np.sqrt((blob_cy - local_cy) ** 2 + (blob_cx - local_cx) ** 2)
            # 用等效半径（由面积反推）替代外接圆半径
            # 六边形细胞的外接圆半径明显大于实际尺度，会导致位置判断偏移
            eff_r = np.sqrt(cell_area / np.pi)
            rel_pos = blob_dist / eff_r if eff_r > 0 else 1.0

            # 按位置选择面积门槛
            if rel_pos < settings['bad_blob_center_radius']:
                hole_thr = settings['bad_hole_area_center']
            else:
                hole_thr = settings['bad_hole_area_edge']

            blob_ratio = blob_area / cell_area
            if blob_ratio > hole_thr:
                if blob_ratio > max_blob_ratio:
                    max_blob_ratio = blob_ratio
                    max_blob_angle_span = blob_angle_span
                    max_blob_pos = rel_pos
                blob_count += 1

    if max_blob_ratio > settings['bad_broken_area']:
        reasons.append('BROKEN')
    elif max_blob_ratio > 0:
        reasons.append('HOLE')

    is_bad = len(reasons) > 0
    return is_bad, reasons, max_blob_ratio, blob_count, max_blob_angle_span, max_blob_pos, dark_thr



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
                is_bad, reasons, max_blob_ratio, blob_count, blob_angle_span, blob_pos, dark_thr = _detect_defects(
                    ys0, xs0, gray, c["er"],
                    (ly0, lx0, ly1, lx1), H_img, W_img, settings)
                bad_reason = '+'.join(reasons)
            else:
                is_bad, max_blob_ratio, blob_count, blob_angle_span, blob_pos, dark_thr, bad_reason = False, 0.0, 0, 0, 0.0, 0.0, ""
            cell_list.append({
                "brightness": peak, "pos": (cx, cy),
                "contours": c["contours"], "mask": c["mask"], "er": c["er"],
                "is_bad": is_bad, "max_blob_ratio": max_blob_ratio,
                "blob_count": blob_count, "blob_angle_span": blob_angle_span,
                "blob_pos": blob_pos, "dark_thr": dark_thr, "reason": bad_reason,
            })

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
    scale = settings['resize_scale']
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        work_image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  缩放: {orig_w}x{orig_h} → {new_w}x{new_h} (倍率={scale})", flush=True)
    else:
        work_image = raw_image
        scale = 1.0

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
    )[0]
    if masks.shape != (H_img, W_img):
        masks = cv2.resize(masks, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    cell_list = _filter_and_rank_mask(masks, work_image, settings)
    total_detected = len(np.unique(masks)) - 1

    thickness  = settings['contour_thickness']
    font_scale = settings['font_scale']

    # ── 标注结果图 ────────────────────────────────────────────────────────────
    res_img = work_image.copy()
    bad_cells    = [c for c in cell_list if c.get('is_bad', False)]
    normal_cells = [c for c in cell_list if not c.get('is_bad', False)]

    # 最亮排行分组（排行开启时按亮度排序，否则保持原序）
    if settings['enable_top_ranking']:
        sorted_normal = sorted(normal_cells, key=lambda x: x["brightness"],
                               reverse=bool(settings['sort_descending']))
        top_cells    = sorted_normal[:top_n]
        other_normal = sorted_normal[top_n:]
    else:
        top_cells    = []
        other_normal = normal_cells

    # 破损排行（按暗块面积降序）
    bad_sorted = sorted(bad_cells, key=lambda x: x.get('max_blob_ratio', 0), reverse=True)

    if cell_list:
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

        # 红色：坏细胞（始终标注，不受排行开关影响）
        if settings['enable_bad_detection']:
            for cell in bad_cells:
                cx, cy    = cell["pos"]
                bad_label = cell.get('reason') or 'BAD'
                cv2.drawContours(res_img, cell["contours"], -1, (0, 0, 255), thickness)
                cv2.putText(res_img, bad_label, (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 黄色：非 top 正常细胞
        idx_start = top_n + 1 if settings['enable_top_ranking'] else 1
        for idx, cell in enumerate(other_normal, start=idx_start):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), thickness)
            cv2.putText(res_img, str(idx), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # 绿色：top_n 最亮正常细胞
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

    # 最亮细胞裁剪图
    if settings['enable_top_ranking']:
        top_crop_dir = os.path.join(save_dir, "crop", "top")
        os.makedirs(top_crop_dir, exist_ok=True)
        for rank, cell in enumerate(top_cells, start=1):
            cx, cy = cell["pos"]
            pad = int(cell["er"] * settings['crop_pad'])
            y1 = max(0, cy - pad); y2 = min(H_img, cy + pad)
            x1 = max(0, cx - pad); x2 = min(W_img, cx + pad)
            crop = res_img[y1:y2, x1:x2].copy()
            cv2.putText(crop, f"Top {rank}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imencode('.png', crop)[1].tofile(
                os.path.join(top_crop_dir, f"top{rank}.png"))
            print(f"  Top{rank} 裁剪图已保存", flush=True)

    # 破损细胞裁剪图
    if settings['enable_bad_ranking'] and settings['enable_bad_detection']:
        bad_crop_dir = os.path.join(save_dir, "crop", "bad")
        os.makedirs(bad_crop_dir, exist_ok=True)
        for rank, cell in enumerate(bad_sorted, start=1):
            cx, cy = cell["pos"]
            pad = int(cell["er"] * settings['crop_pad'])
            y1 = max(0, cy - pad); y2 = min(H_img, cy + pad)
            x1 = max(0, cx - pad); x2 = min(W_img, cx + pad)
            crop = res_img[y1:y2, x1:x2].copy()
            label = f"Bad {rank} | {cell.get('reason','BAD')}"
            cv2.putText(crop, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imencode('.png', crop)[1].tofile(
                os.path.join(bad_crop_dir, f"bad{rank}.png"))
        print(f"  破损细胞裁剪图: {len(bad_sorted)} 张", flush=True)

    csv_header = "编号,状态,标记原因,直径(px),坐标X,坐标Y,亮度值,最大暗块比例,暗块数量,暗块角度覆盖,暗块相对位置,实际暗块阈值\n"
    with open(os.path.join(save_dir, f"{stem}_data.csv"), 'w', encoding='utf-8') as f:
        f.write(f"# 坐标系: 缩放后图片 ({W_img}x{H_img})\n")
        f.write(csv_header)
        for i, cell in enumerate(cell_list, start=1):
            cx, cy          = cell["pos"]
            diameter        = round(cell["er"] * 2, 1)
            status          = '!!!BAD!!!' if cell.get('is_bad', False) else 'OK'
            reason          = cell.get('reason', '') or '-'
            max_blob_ratio  = round(cell.get('max_blob_ratio', 0.0), 4)
            blob_count      = cell.get('blob_count', 0)
            blob_angle_span = cell.get('blob_angle_span', 0)
            blob_pos        = round(cell.get('blob_pos', 0.0), 3)
            dark_thr        = round(cell.get('dark_thr', 0.0), 1)
            f.write(f"{i},{status},{reason},{diameter},{cx},{cy},{int(cell['brightness'])},{max_blob_ratio},{blob_count},{blob_angle_span},{blob_pos},{dark_thr}\n")

    with open(os.path.join(save_dir, f"{stem}_data_original.csv"), 'w', encoding='utf-8') as f:
        f.write(f"# 坐标系: 原图 ({orig_w}x{orig_h}), scale={scale:.4f}\n")
        f.write(csv_header)
        for i, cell in enumerate(cell_list, start=1):
            cx, cy          = cell["pos"]
            orig_cx         = int(cx / scale)
            orig_cy         = int(cy / scale)
            orig_diameter   = round(cell["er"] * 2 / scale, 1)
            status          = '!!!BAD!!!' if cell.get('is_bad', False) else 'OK'
            reason          = cell.get('reason', '') or '-'
            max_blob_ratio  = round(cell.get('max_blob_ratio', 0.0), 4)
            blob_count      = cell.get('blob_count', 0)
            blob_angle_span = cell.get('blob_angle_span', 0)
            blob_pos        = round(cell.get('blob_pos', 0.0), 3)
            dark_thr        = round(cell.get('dark_thr', 0.0), 1)
            f.write(f"{i},{status},{reason},{orig_diameter},{orig_cx},{orig_cy},{int(cell['brightness'])},{max_blob_ratio},{blob_count},{blob_angle_span},{blob_pos},{dark_thr}\n")

    if settings['enable_bad_ranking'] and settings['enable_bad_detection']:
        cell_index = {id(c): i + 1 for i, c in enumerate(cell_list)}
        with open(os.path.join(save_dir, f"{stem}_bad.csv"), 'w', encoding='utf-8') as f:
            f.write("破损排名,原编号,标记类型,暗块面积比例,直径(px),坐标X,坐标Y,亮度值,暗块相对位置\n")
            for rank, cell in enumerate(bad_sorted, start=1):
                orig_idx    = cell_index.get(id(cell), -1)
                cx, cy      = cell["pos"]
                diameter    = round(cell["er"] * 2, 1)
                reason      = cell.get('reason', '') or '-'
                blob_ratio  = round(cell.get('max_blob_ratio', 0.0), 4)
                blob_pos    = round(cell.get('blob_pos', 0.0), 3)
                f.write(f"{rank},{orig_idx},{reason},{blob_ratio},{diameter},{cx},{cy},{int(cell['brightness'])},{blob_pos}\n")

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
    print(f"  diameter={settings['diameter']}  top_n={settings['top_n']}  resize_scale={settings['resize_scale']}", flush=True)

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
