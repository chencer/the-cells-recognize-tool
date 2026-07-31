"""细胞图像识别与排行工具（CellAppCP3）。

功能:
    - 读取输入目录下的图像，通过 Cellpose 模型进行细胞分割。
    - 应用完整度、面积、圆形度多级过滤，计算亮度与缺陷状态。
    - 输出标注结果图、局部裁剪图与多份 CSV 数据表。

用法:
    将待处理图片放入 input 目录，执行 ``python cells_find.py``。
    结果保存至 results/{图片名}/ 目录。

配置:
    全部参数通过 settings.txt 管理，缺失项使用内置默认值。
"""

import math
import os
import ssl
import sys
import time

import cv2
import numpy as np
import torch
from cellpose import models as cp_models


# ---------------------------------------------------------------------------
# 绘图常量
# ---------------------------------------------------------------------------
COLOR_TOP    = (0, 255, 0)      # BGR 绿色，用于最亮细胞轮廓
COLOR_NORMAL = (0, 255, 255)    # BGR 黄色，用于普通细胞轮廓
COLOR_BAD    = (0, 0, 255)      # BGR 红色，用于缺陷细胞轮廓
COLOR_LABEL  = (220, 220, 220)  # BGR 浅灰，用于编号文字

RING_INNER_RATIO = 0.8          # 压暗环内边界占细胞半径的比例
RING_DIM_FACTOR  = 0.6          # 压暗环区域的亮度衰减系数
CENTER_DOT_R     = 5            # 最亮细胞中心标记点半径，单位像素
LABEL_OFFSET     = (6, -6)      # 编号文字相对细胞中心的偏移，单位像素
CROP_LABEL_POS   = (10, 30)     # 裁剪图内标签的绘制位置，单位像素
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# CSV 表头与字段定义
# ---------------------------------------------------------------------------
DATA_HEADER = "编号,状态,标记原因,直径(px),坐标X,坐标Y,亮度值,最大暗块比例,暗块数量,暗块角度覆盖,暗块相对位置,实际暗块阈值"
TOP_HEADER  = "亮度排名,原编号,直径(px),坐标X,坐标Y,亮度值"
BAD_HEADER  = "破损排名,原编号,标记类型,暗块面积比例,直径(px),坐标X,坐标Y,亮度值,暗块相对位置"
TOP_FIELDS  = ['rank', 'idx', 'diameter', 'cx', 'cy', 'brightness']
BAD_FIELDS  = ['rank', 'idx', 'reason', 'blob_ratio', 'diameter', 'cx', 'cy', 'brightness', 'blob_pos']


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def get_resource_path(relative_path):
    """解析资源文件路径，兼容 PyInstaller 打包环境。

    Args:
        relative_path: 相对于程序根目录的路径。

    Returns:
        资源文件的绝对路径。
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_settings():
    """从 settings.txt 加载配置参数。

    配置文件或单项参数缺失时，回退至内置默认值。数值类型依据
    默认值推断，字符串参数保持原样。

    Returns:
        dict: 完整配置字典，键为参数名。
    """
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


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model(settings):
    """加载 Cellpose 模型并选定计算设备。

    优先加载程序目录下的同名模型文件，该文件不存在时使用
    Cellpose 内置模型。设备按 CUDA、MPS、CPU 的优先级选择。

    Args:
        settings: 配置字典，需包含 model_name 键。

    Returns:
        CellposeModel: 已加载的模型实例。
    """
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


# ---------------------------------------------------------------------------
# 缺陷检测
# ---------------------------------------------------------------------------
def _detect_defects(gy, gx, gray, er, bbox, H_img, W_img, settings):
    """检测单个细胞内部的暗区缺陷。

    通过连通域分析定位暗块，依次应用紧凑度、角度覆盖、位置分级
    三重判据，排除细胞边缘的正常暗环，仅保留真实缺陷。

    Args:
        gy: 细胞像素的行坐标数组。
        gx: 细胞像素的列坐标数组。
        gray: 全图灰度图。
        er: 细胞最小外接圆半径。
        bbox: 细胞外接矩形，格式为 (top, left, bottom, right)。
        H_img: 图像高度。
        W_img: 图像宽度。
        settings: 配置字典。

    Returns:
        tuple: 依次为是否存在缺陷、缺陷类型列表、最大暗块面积比、
            暗块数量、角度覆盖度、暗块相对位置、实际使用的阈值。
    """
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

            # 采用等效半径归一化位置，避免六边形细胞使用外接圆半径时的偏移
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


# ---------------------------------------------------------------------------
# 过滤链
# ---------------------------------------------------------------------------
def _filter_and_rank_mask(masks, raw_image, settings):
    """对分割结果执行多级过滤并计算细胞属性。

    过滤顺序依次为完整度、面积、圆形度。通过筛选的细胞将计算
    亮度值与缺陷状态，并分配稳定编号。

    Args:
        masks: Cellpose 输出的标签矩阵。
        raw_image: 工作图像，BGR 格式。
        settings: 配置字典。

    Returns:
        list[dict]: 有效细胞列表，每项包含 idx、pos、er、
            brightness、contours、mask、is_bad 等字段。
    """
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

    # Step4: 亮度计算与缺陷检测
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

    # 分配一次性的稳定编号，供后续 CSV 与排行表交叉引用
    for i, cell in enumerate(cell_list, start=1):
        cell['idx'] = i

    print(f"  [Step4] 有效细胞: {len(cell_list)}", flush=True)
    return cell_list


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------
def _load_and_scale(image_path, settings):
    """读取图像并按配置倍率缩放。

    Args:
        image_path: 图像文件路径。
        settings: 配置字典，需包含 resize_scale 键。

    Returns:
        tuple: 依次为原图、工作图、原始尺寸元组、实际缩放倍率。
            读取失败时返回四个 None。
    """
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


# ---------------------------------------------------------------------------
# 细胞分组
# ---------------------------------------------------------------------------
def _group_cells(cell_list, settings):
    """按缺陷状态与亮度排行对细胞分组。

    缺陷细胞按暗块面积降序排列。亮度排行关闭时，最亮组为空，
    全部正常细胞归入其余组。

    Args:
        cell_list: 有效细胞列表。
        settings: 配置字典。

    Returns:
        tuple: 依次为缺陷细胞列表、亮度排行前 N 项、其余正常细胞。
    """
    bad    = [c for c in cell_list if c.get('is_bad')]
    normal = [c for c in cell_list if not c.get('is_bad')]

    bad_sorted = sorted(bad, key=lambda c: c.get('max_blob_ratio', 0), reverse=True)

    if settings['enable_top_ranking']:
        ranked = sorted(normal, key=lambda c: c['brightness'],
                        reverse=bool(settings['sort_descending']))
        top_n = settings['top_n']
        return bad_sorted, ranked[:top_n], ranked[top_n:]

    return bad_sorted, [], normal


# ---------------------------------------------------------------------------
# 结果图标注
# ---------------------------------------------------------------------------
def _annotate(work_image, cell_list, bad_cells, top_cells, other_cells, settings):
    """在工作图上绘制细胞轮廓与标签。

    绘制顺序依次为外圈压暗遮罩、缺陷细胞（红）、普通细胞（黄）、
    最亮细胞（绿）。后绘制的图层覆盖先绘制的图层。

    Args:
        work_image: 工作图像，BGR 格式。
        cell_list: 全部有效细胞。
        bad_cells: 缺陷细胞列表。
        top_cells: 亮度排行前 N 项。
        other_cells: 其余正常细胞列表。
        settings: 配置字典。

    Returns:
        numpy.ndarray: 标注后的图像副本。
    """
    res_img = work_image.copy()
    if not cell_list:
        return res_img

    H_img, W_img = res_img.shape[:2]
    thickness    = settings['contour_thickness']
    font_scale   = settings['font_scale']
    dx, dy       = LABEL_OFFSET

    # 限制在外接矩形范围内运算，避免为每个细胞分配全图尺寸的临时数组
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

    # 缺陷细胞始终标注，不受排行开关影响
    if settings['enable_bad_detection']:
        for cell in bad_cells:
            cx, cy = cell["pos"]
            label  = cell.get('reason') or 'BAD'
            cv2.drawContours(res_img, cell["contours"], -1, COLOR_BAD, thickness)
            cv2.putText(res_img, label, (cx + dx, cy + dy), FONT, 0.5, COLOR_BAD, 1)

    # 未进入最亮排行的正常细胞
    idx_start = settings['top_n'] + 1 if settings['enable_top_ranking'] else 1
    for idx, cell in enumerate(other_cells, start=idx_start):
        cx, cy = cell["pos"]
        cv2.drawContours(res_img, cell["contours"], -1, COLOR_NORMAL, thickness)
        cv2.putText(res_img, str(idx), (cx + dx, cy + dy), FONT, 0.5, COLOR_LABEL, 1)

    # 亮度排行前 N 项，最后绘制以保证覆盖前面的黄色描边
    for rank, cell in enumerate(top_cells, start=1):
        cx, cy = cell["pos"]
        cv2.drawContours(res_img, cell["contours"], -1, COLOR_TOP, thickness)
        cv2.circle(res_img, (cx, cy), CENTER_DOT_R, COLOR_TOP, -1)
        cv2.putText(res_img, f"{rank} | {int(cell['brightness'])}",
                    (cx + dx + 2, cy + dy - 2), FONT, font_scale, COLOR_TOP, 2)

    return res_img


# ---------------------------------------------------------------------------
# 裁剪图输出
# ---------------------------------------------------------------------------
def _save_crops(res_img, cells, out_dir, label_fn, file_prefix, color, settings):
    """批量输出细胞局部裁剪图。

    Args:
        res_img: 已标注的结果图。
        cells: 待裁剪的细胞列表。
        out_dir: 输出目录，不存在时自动创建。
        label_fn: 标签生成函数，签名为 (rank, cell) -> str。
        file_prefix: 输出文件名前缀。
        color: 标签文字颜色，BGR 格式。
        settings: 配置字典，需包含 crop_pad 键。

    Returns:
        int: 实际输出的图片数量。
    """
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
    """依据配置开关输出最亮细胞与缺陷细胞的裁剪图。

    Args:
        res_img: 已标注的结果图。
        save_dir: 结果目录。
        top_cells: 亮度排行前 N 项。
        bad_sorted: 已排序的缺陷细胞列表。
        settings: 配置字典。
    """
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


# ---------------------------------------------------------------------------
# CSV 输出
# ---------------------------------------------------------------------------
def _write_csv(path, header, rows, comment=None):
    """写入 CSV 文件，采用 UTF-8 BOM 编码以兼容 Excel。

    Args:
        path: 输出文件路径。
        header: 表头行，不含换行符。
        rows: 已格式化的数据行列表。
        comment: 可选的首行注释，写入时自动添加井号前缀。
    """
    with open(path, 'w', encoding='utf-8-sig') as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")


def _fmt_data_row(cell, scale=1.0):
    """格式化完整数据表的单行记录。

    Args:
        cell: 细胞数据字典。
        scale: 缩放倍率。取值 1.0 时输出工作图坐标，
            其他取值时换算为原图坐标。

    Returns:
        str: 以逗号分隔的数据行。
    """
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
    """按指定字段顺序格式化排行表的单行记录。

    Args:
        rank: 排名序号。
        cell: 细胞数据字典。
        fields: 字段名列表，决定输出列的顺序。
        scale: 缩放倍率，含义与 _fmt_data_row 相同。

    Returns:
        str: 以逗号分隔的数据行。
    """
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
    """依据配置开关输出全部 CSV 文件。

    完整数据表固定输出两份，分别采用工作图坐标与原图坐标。
    排行表由对应开关决定是否输出。

    Args:
        save_dir: 结果目录。
        stem: 图片文件名，不含扩展名。
        cell_list: 全部有效细胞。
        top_cells: 亮度排行前 N 项。
        bad_sorted: 已排序的缺陷细胞列表。
        scaled_note: 工作图坐标系说明文字。
        orig_note: 原图坐标系说明文字。
        scale: 缩放倍率。
        settings: 配置字典。
    """
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


# ---------------------------------------------------------------------------
# 单张图片处理
# ---------------------------------------------------------------------------
def process_image(model, image_path, results_dir, settings):
    """处理单张图片的完整流程。

    流程依次为读图缩放、模型分割、多级过滤、细胞分组、结果标注、
    裁剪图导出与 CSV 导出。

    Args:
        model: 已加载的 Cellpose 模型。
        image_path: 待处理图片路径。
        results_dir: 结果根目录。
        settings: 配置字典。
    """
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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    """程序入口，扫描 input 目录并批量处理图片。"""
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
