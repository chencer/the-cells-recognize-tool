import os
import cv2
import torch
import numpy as np
from cellpose import models as cp_models
import sys
import ssl
import math
import time


# ── 绘图常量 ──────────────────────────────────────────────────────────────
COLOR_TOP    = (0, 255, 0)      # 绿色：最亮细胞
COLOR_NORMAL = (0, 255, 255)    # 黄色：普通细胞
COLOR_BAD    = (0, 0, 255)      # 红色：破损细胞
COLOR_LABEL  = (220, 220, 220)  # 浅灰：编号文字

RING_INNER_RATIO = 0.8   # 压暗环内圈半径比例
RING_DIM_FACTOR  = 0.6   # 压暗环亮度系数
CENTER_DOT_R     = 5     # 最亮细胞中心点半径
LABEL_OFFSET     = (6, -6)    # 编号文字相对细胞中心偏移
CROP_LABEL_POS   = (10, 30)   # 裁剪图标签位置
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ── CSV 表头与字段定义 ────────────────────────────────────────────────────
DATA_HEADER = "编号,状态,标记原因,直径(px),坐标X,坐标Y,亮度值,最大暗块比例,暗块数量,暗块角度覆盖,暗块相对位置,实际暗块阈值"
TOP_HEADER  = "亮度排名,原编号,直径(px),坐标X,坐标Y,亮度值"
BAD_HEADER  = "破损排名,原编号,标记类型,暗块面积比例,直径(px),坐标X,坐标Y,亮度值,暗块相对位置"
TOP_FIELDS  = ['rank', 'idx', 'diameter', 'cx', 'cy', 'brightness']
BAD_FIELDS  = ['rank', 'idx', 'reason', 'blob_ratio', 'diameter', 'cx', 'cy', 'brightness', 'blob_pos']


def get_resource_path(relative_path):
    """PyInstaller 打包资源路径解析"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# --- 配置加载 ---
def load_settings():
    """从 settings.txt 读取参数，缺失键用默认值兜底"""
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
    """加载 Cellpose 模型，本地文件优先，否则用内置模型"""
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


# --- 坏细胞检测 ---
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

    if settings['bad_adaptive_threshold']:
        cy_c = float(np.mean(gy))
        cx_c = float(np.mean(gx))
        dist_all = np.sqrt((gy - cy_c) ** 2 + (gx - cx_c) ** 2)
        eff_r_full = np.sqrt(cell_area / np.pi)
        inner_sel = dist_all <= eff_r_full * settings['bad_adaptive_inner']
        if np.sum(inner_sel) > 10:
            inner_vals = gray[gy[inner_sel], gx[inner_sel]]
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
                angles = np.arctan2(blob_pixels[:, 0] - local_cy,
                                    blob_pixels[:, 1] - local_cx)
                angles_deg = np.degrees((angles + 2 * np.pi) % (2 * np.pi))
                hist, _ = np.histogram(angles_deg, bins=36, range=(0, 360))
                blob_angle_span = int(np.sum(hist > 0)) * 10
                if blob_angle_span >= settings['bad_blob_angle_span']:
                    continue

            # 暗块中心到细胞中心的相对距离（用等效半径避免六边形细胞位置偏移）
            blob_cy = float(centroids[i][1])
            blob_cx = float(centroids[i][0])
            blob_dist = np.sqrt((blob_cy - local_cy) ** 2 + (blob_cx - local_cx) ** 2)
            eff_r = np.sqrt(cell_area / np.pi)
            rel_pos = blob_dist / eff_r if eff_r > 0 else 1.0

            hole_thr = (settings['bad_hole_area_center']
                        if rel_pos < settings['bad_blob_center_radius']
                        else settings['bad_hole_area_edge'])

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
    """完整度/面积/圆形度/亮度四步过滤，返回有效细胞列表（带稳定 idx）"""
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
        })
    print(f"  [Step1] 完整度过滤: {len(candidates)} / {total_detected}", flush=True)

    # Step2: 面积过滤
    if candidates:
        median_area = float(np.median([c["mask_area"] for c in candidates]))
        candidates  = [c for c in candidates if c["mask_area"] >= median_area * area_ratio]
    print(f"  [Step2] 面积过滤: {len(candidates)}", flush=True)

    # Step3: 圆形度过滤
    candidates = [c for c in candidates
                  if c["perimeter"] > 0 and
                     (4 * math.pi * c["mask_area"] / (c["perimeter"] ** 2)) >= circ_thr]
    print(f"  [Step3] 圆形度过滤: {len(candidates)}", flush=True)

    # Step4: 亮度计算 + 坏细胞检测
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
        inner_mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
        cv2.circle(inner_mask, (cx - rx1, cy - ry1), ir, 1, -1)
        sample_mask = (inner_mask > 0) & (c["mask"][ry1:ry2, rx1:rx2] > 0)
        cell_pixels = gray[ry1:ry2, rx1:rx2][sample_mask]
        if len(cell_pixels) <= min_pixels:
            continue
        k    = max(1, int(len(cell_pixels) * top_pct))
        peak = float(np.mean(np.partition(cell_pixels, -k)[-k:]))

        if settings['enable_bad_detection']:
            ys0, xs0 = np.where(c["mask"] > 0)
            bbox = (int(ys0.min()), int(xs0.min()), int(ys0.max()), int(xs0.max()))
            is_bad, reasons, mbr, bc, bas, bp, dth = _detect_defects(
                ys0, xs0, gray, c["er"], bbox, H_img, W_img, settings)
            bad_reason = '+'.join(reasons)
        else:
            is_bad, mbr, bc, bas, bp, dth, bad_reason = False, 0.0, 0, 0, 0.0, 0.0, ""

        cell_list.append({
            "brightness": peak, "pos": (cx, cy),
            "contours": c["contours"], "mask": c["mask"], "er": c["er"],
            "is_bad": is_bad, "max_blob_ratio": mbr,
            "blob_count": bc, "blob_angle_span": bas,
            "blob_pos": bp, "dark_thr": dth, "reason": bad_reason,
        })

    # 分配稳定编号，供后续 CSV / 排行引用
    for i, cell in enumerate(cell_list, start=1):
        cell['idx'] = i

    print(f"  [Step4] 有效细胞: {len(cell_list)}", flush=True)
    return cell_list


# --- 图像预处理 ---
def _load_and_scale(image_path, settings):
    """读取图片并按倍率缩放。返回 (原图, 工作图, 原尺寸, 缩放倍率)"""
    raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        return None, None, None, None

    orig_h, orig_w = raw.shape[:2]
    scale = settings['resize_scale']
    if scale < 1.0:
        work = cv2.resize(raw, (int(orig_w * scale), int(orig_h * scale)),
                          interpolation=cv2.INTER_AREA)
        print(f"  缩放: {orig_w}x{orig_h} → {work.shape[1]}x{work.shape[0]} "
              f"(倍率={scale})", flush=True)
    else:
        work, scale = raw, 1.0

    return raw, work, (orig_w, orig_h), scale


# --- 细胞分组 ---
def _group_cells(cell_list, settings):
    """按状态和排行分组。返回 (破损排序, 最亮top, 其余正常)"""
    bad    = [c for c in cell_list if c.get('is_bad')]
    normal = [c for c in cell_list if not c.get('is_bad')]

    bad_sorted = sorted(bad, key=lambda c: c.get('max_blob_ratio', 0), reverse=True)

    if settings['enable_top_ranking']:
        ranked = sorted(normal, key=lambda c: c['brightness'],
                        reverse=bool(settings['sort_descending']))
        top_n = settings['top_n']
        return bad_sorted, ranked[:top_n], ranked[top_n:]

    return bad_sorted, [], normal


# --- 结果图标注 ---
def _annotate(work_image, cell_list, bad_cells, top_cells, other_cells, settings):
    """绘制压暗遮罩 + 三色描边 + 编号，返回标注后的图"""
    res_img = work_image.copy()
    if not cell_list:
        return res_img

    H_img, W_img = res_img.shape[:2]
    thickness    = settings['contour_thickness']
    font_scale   = settings['font_scale']
    dx, dy       = LABEL_OFFSET

    # 压暗环遮罩（只在细胞局部范围运算，避免全图数组分配）
    for cell in cell_list:
        cx, cy = cell["pos"]
        er_int = max(1, int(cell["er"]))
        ir_int = max(1, int(cell["er"] * RING_INNER_RATIO))
        y1 = max(0, cy - er_int); y2 = min(H_img, cy + er_int + 1)
        x1 = max(0, cx - er_int); x2 = min(W_img, cx + er_int + 1)
        lh, lw = y2 - y1, x2 - x1
        if lh <= 0 or lw <= 0:
            continue
        outer_m = np.zeros((lh, lw), dtype=np.uint8)
        inner_m = np.zeros((lh, lw), dtype=np.uint8)
        cv2.circle(outer_m, (cx - x1, cy - y1), er_int, 1, -1)
        cv2.circle(inner_m, (cx - x1, cy - y1), ir_int, 1, -1)
        ring = (outer_m > 0) & (inner_m == 0) & (cell["mask"][y1:y2, x1:x2] > 0)
        region = res_img[y1:y2, x1:x2]
        region[ring] = (region[ring] * RING_DIM_FACTOR).astype(np.uint8)

    # 红色：破损细胞（始终标注，不受排行开关影响）
    if settings['enable_bad_detection']:
        for cell in bad_cells:
            cx, cy = cell["pos"]
            label  = cell.get('reason') or 'BAD'
            cv2.drawContours(res_img, cell["contours"], -1, COLOR_BAD, thickness)
            cv2.putText(res_img, label, (cx + dx, cy + dy), FONT, 0.5, COLOR_BAD, 1)

    # 黄色：非 top 正常细胞
    idx_start = settings['top_n'] + 1 if settings['enable_top_ranking'] else 1
    for idx, cell in enumerate(other_cells, start=idx_start):
        cx, cy = cell["pos"]
        cv2.drawContours(res_img, cell["contours"], -1, COLOR_NORMAL, thickness)
        cv2.putText(res_img, str(idx), (cx + dx, cy + dy), FONT, 0.5, COLOR_LABEL, 1)

    # 绿色：top_n 最亮正常细胞
    for rank, cell in enumerate(top_cells, start=1):
        cx, cy = cell["pos"]
        cv2.drawContours(res_img, cell["contours"], -1, COLOR_TOP, thickness)
        cv2.circle(res_img, (cx, cy), CENTER_DOT_R, COLOR_TOP, -1)
        cv2.putText(res_img, f"{rank} | {int(cell['brightness'])}",
                    (cx + dx + 2, cy + dy - 2), FONT, font_scale, COLOR_TOP, 2)

    return res_img


# --- 裁剪图输出 ---
def _save_crops(res_img, cells, out_dir, label_fn, file_prefix, color, settings):
    """通用裁剪图输出。label_fn(rank, cell) 返回图上标签文字"""
    if not cells:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    H, W = res_img.shape[:2]

    for rank, cell in enumerate(cells, start=1):
        cx, cy = cell["pos"]
        pad = int(cell["er"] * settings['crop_pad'])
        y1, y2 = max(0, cy - pad), min(H, cy + pad)
        x1, x2 = max(0, cx - pad), min(W, cx + pad)
        crop = res_img[y1:y2, x1:x2].copy()
        cv2.putText(crop, label_fn(rank, cell), CROP_LABEL_POS, FONT, 0.8, color, 2)
        cv2.imencode('.png', crop)[1].tofile(
            os.path.join(out_dir, f"{file_prefix}{rank}.png"))

    return len(cells)


def _save_all_crops(res_img, save_dir, top_cells, bad_sorted, settings):
    """输出最亮和破损细胞的裁剪图"""
    if settings['enable_top_ranking']:
        n = _save_crops(res_img, top_cells,
                        os.path.join(save_dir, "crop", "top"),
                        lambda r, c: f"Top {r}", "top", COLOR_TOP, settings)
        print(f"  最亮细胞裁剪图: {n} 张", flush=True)

    if settings['enable_bad_ranking'] and settings['enable_bad_detection']:
        n = _save_crops(res_img, bad_sorted,
                        os.path.join(save_dir, "crop", "bad"),
                        lambda r, c: f"Bad {r} | {c.get('reason', 'BAD')}",
                        "bad", COLOR_BAD, settings)
        print(f"  破损细胞裁剪图: {n} 张", flush=True)


# --- CSV 输出 ---
def _write_csv(path, header, rows, comment=None):
    """通用 CSV 写入。rows 是已格式化好的字符串列表"""
    with open(path, 'w', encoding='utf-8-sig') as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")


def _fmt_data_row(cell, scale=1.0):
    """格式化 data.csv 行。scale=1.0 为缩放坐标，否则换算回原图"""
    cx, cy = cell["pos"]
    if scale != 1.0:
        cx, cy = int(cx / scale), int(cy / scale)
        diameter = round(cell["er"] * 2 / scale, 1)
    else:
        diameter = round(cell["er"] * 2, 1)

    return ",".join([
        str(cell['idx']),
        '!!!BAD!!!' if cell.get('is_bad') else 'OK',
        cell.get('reason') or '-',
        str(diameter), str(cx), str(cy),
        str(int(cell['brightness'])),
        str(round(cell.get('max_blob_ratio', 0.0), 4)),
        str(cell.get('blob_count', 0)),
        str(cell.get('blob_angle_span', 0)),
        str(round(cell.get('blob_pos', 0.0), 3)),
        str(round(cell.get('dark_thr', 0.0), 1)),
    ])


def _fmt_rank_row(rank, cell, fields, scale=1.0):
    """按 fields 顺序格式化排行 CSV 行"""
    cx, cy = cell["pos"]
    if scale != 1.0:
        cx, cy = int(cx / scale), int(cy / scale)
        diameter = round(cell["er"] * 2 / scale, 1)
    else:
        diameter = round(cell["er"] * 2, 1)

    base = {
        'rank': rank, 'idx': cell['idx'], 'diameter': diameter,
        'cx': cx, 'cy': cy, 'brightness': int(cell['brightness']),
        'reason': cell.get('reason') or '-',
        'blob_ratio': round(cell.get('max_blob_ratio', 0.0), 4),
        'blob_pos': round(cell.get('blob_pos', 0.0), 3),
    }
    return ",".join(str(base[f]) for f in fields)


def _write_all_csvs(save_dir, stem, cell_list, top_cells, bad_sorted,
                    scaled_note, orig_note, scale, settings):
    """输出全部 CSV：data / data_original / top / top_original / bad / bad_original"""
    _write_csv(os.path.join(save_dir, f"{stem}_data.csv"), DATA_HEADER,
               [_fmt_data_row(c) for c in cell_list], scaled_note)
    _write_csv(os.path.join(save_dir, f"{stem}_data_original.csv"), DATA_HEADER,
               [_fmt_data_row(c, scale) for c in cell_list], orig_note)

    if settings['enable_top_ranking']:
        _write_csv(os.path.join(save_dir, f"{stem}_top.csv"), TOP_HEADER,
                   [_fmt_rank_row(r, c, TOP_FIELDS)
                    for r, c in enumerate(top_cells, 1)], scaled_note)
        _write_csv(os.path.join(save_dir, f"{stem}_top_original.csv"), TOP_HEADER,
                   [_fmt_rank_row(r, c, TOP_FIELDS, scale)
                    for r, c in enumerate(top_cells, 1)], orig_note)
        print(f"  最亮排行 CSV: {len(top_cells)} 条", flush=True)

    if settings['enable_bad_ranking'] and settings['enable_bad_detection']:
        _write_csv(os.path.join(save_dir, f"{stem}_bad.csv"), BAD_HEADER,
                   [_fmt_rank_row(r, c, BAD_FIELDS)
                    for r, c in enumerate(bad_sorted, 1)], scaled_note)
        _write_csv(os.path.join(save_dir, f"{stem}_bad_original.csv"), BAD_HEADER,
                   [_fmt_rank_row(r, c, BAD_FIELDS, scale)
                    for r, c in enumerate(bad_sorted, 1)], orig_note)
        print(f"  破损排行 CSV: {len(bad_sorted)} 条", flush=True)


# --- 单张图片处理（流程编排） ---
def process_image(model, image_path, results_dir, settings):
    """单张图片完整流程：读图 → 分割 → 过滤 → 分组 → 标注 → 输出"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n[{stem}] 处理中...", flush=True)

    raw_image, work_image, orig_size, scale = _load_and_scale(image_path, settings)
    if raw_image is None:
        print("  ❌ 无法读取图片，跳过", flush=True)
        return

    orig_w, orig_h = orig_size
    H_img, W_img   = work_image.shape[:2]
    save_dir       = os.path.join(results_dir, stem)
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

    total_detected = len(np.unique(masks)) - 1
    cell_list      = _filter_and_rank_mask(masks, work_image, settings)
    bad_sorted, top_cells, other_cells = _group_cells(cell_list, settings)

    res_img = _annotate(work_image, cell_list, bad_sorted,
                        top_cells, other_cells, settings)
    print("  标注完成", flush=True)

    print("  保存中...", flush=True)
    cv2.imencode('.png', res_img)[1].tofile(
        os.path.join(save_dir, f"{stem}_result.png"))

    _save_all_crops(res_img, save_dir, top_cells, bad_sorted, settings)

    scaled_note = f"坐标系: 缩放后图片 ({W_img}x{H_img})"
    orig_note   = f"坐标系: 原图 ({orig_w}x{orig_h}), scale={scale:.4f}"
    _write_all_csvs(save_dir, stem, cell_list, top_cells, bad_sorted,
                    scaled_note, orig_note, scale, settings)

    summary = f"  检测: {total_detected}  有效: {len(cell_list)}"
    if bad_sorted:
        summary += f"  破损: {len(bad_sorted)}"
    if top_cells:
        cx1, cy1 = top_cells[0]["pos"]
        summary += f"  最亮 ({cx1}, {cy1}) 亮度={int(top_cells[0]['brightness'])}"
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
    print(f"  diameter={settings['diameter']}  top_n={settings['top_n']}  "
          f"resize_scale={settings['resize_scale']}", flush=True)

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
