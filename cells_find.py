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
        'top_n': 3, 'large_mode': 1,
        'tile_w': 2048, 'tile_h': 1080, 'tile_overlap': 0.2,
        'crop_pad': 2.0, 'contour_thickness': 2, 'font_scale': 0.7,
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
                        if isinstance(defaults[key], int):
                            defaults[key] = int(val)
                        elif isinstance(defaults[key], float):
                            defaults[key] = float(val)
        print(f"✅ 已读取配置: {path}", flush=True)
    else:
        print(f"⚠️ 未找到 settings.txt，使用默认参数", flush=True)
    return defaults


# --- 模型加载 ---
def load_model():
    print("正在加载模型...", flush=True)
    torch.serialization.add_safe_globals([cp_models.CellposeModel])
    import torch.serialization as _ts
    torch.load = lambda *a, **kw: _ts.load(*a, **kw, weights_only=False)
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

    model_path = get_resource_path("cyto3")
    if os.path.exists(model_path):
        m = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
        print(f"✅ 成功加载本地模型: {model_path}", flush=True)
    else:
        m = cp_models.Cellpose(gpu=use_gpu, model_type="cyto3")
        print("✅ 使用系统默认路径加载 cyto3", flush=True)
    return m


# --- 大图分块识别，返回候选细胞列表 ---
def _tile_and_merge(model, raw_image, settings):
    H, W   = raw_image.shape[:2]
    TW     = settings['tile_w']
    TH     = settings['tile_h']
    OX     = int(TW * settings['tile_overlap'])
    OY     = int(TH * settings['tile_overlap'])
    SX     = TW - OX
    SY     = TH - OY

    def origins(total, tile, step):
        pts = list(range(0, total - tile, step))
        pts.append(max(0, total - tile))
        return sorted(set(pts))

    xs    = origins(W, TW, SX)
    ys    = origins(H, TH, SY)
    total = len(xs) * len(ys)
    print(f"  裁切为 {len(xs)}x{len(ys)} = {total} 块 ({TW}x{TH}, 重叠{int(settings['tile_overlap']*100)}%)", flush=True)

    candidates = []
    count = 0
    for y0 in ys:
        for x0 in xs:
            count += 1
            print(f"  [{count}/{total}] 处理中...", flush=True)
            x1   = min(x0 + TW, W)
            y1   = min(y0 + TH, H)
            tile = raw_image[y0:y1, x0:x1]

            th, tw = tile.shape[:2]
            if th < TH or tw < TW:
                pad = np.zeros((TH, TW, raw_image.shape[2]), dtype=raw_image.dtype)
                pad[:th, :tw] = tile
                tile = pad

            tile_masks = model.eval(
                tile,
                diameter=settings['diameter'],
                channels=[0, 0],
                flow_threshold=settings['flow_threshold'],
                cellprob_threshold=settings['cellprob_threshold'],
                min_size=settings['min_size'],
                niter=settings['niter'],
                resample=False,
                tile=False,
            )[0]

            n_cells = len(np.unique(tile_masks)) - 1
            print(f"    → {n_cells} 个细胞", flush=True)

            for cid in np.unique(tile_masks)[1:]:
                ly, lx = np.where(tile_masks == cid)
                valid  = (ly < (y1 - y0)) & (lx < (x1 - x0))
                gy     = (ly[valid] + y0).astype(np.int32)
                gx     = (lx[valid] + x0).astype(np.int32)
                if len(gy) < 10:
                    continue
                bbox = (int(gy.min()), int(gx.min()), int(gy.max()), int(gx.max()))
                candidates.append({'gy': gy, 'gx': gx, 'area': len(gy), 'bbox': bbox})

    # IoU 去重
    print(f"  合并去重中... ({len(candidates)} 个候选细胞)", flush=True)
    keep = [True] * len(candidates)

    for i in range(len(candidates)):
        if not keep[i]:
            continue
        bi = candidates[i]['bbox']
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            bj = candidates[j]['bbox']
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            iy1 = max(bi[0], bj[0]); ix1 = max(bi[1], bj[1])
            iy2 = min(bi[2], bj[2]); ix2 = min(bi[3], bj[3])
            if iy2 <= iy1 or ix2 <= ix1:
                continue
            rh, rw = iy2 - iy1, ix2 - ix1
            mi = np.zeros((rh, rw), dtype=bool)
            mj = np.zeros((rh, rw), dtype=bool)
            ci = candidates[i]
            vi = (ci['gy'] >= iy1) & (ci['gy'] < iy2) & (ci['gx'] >= ix1) & (ci['gx'] < ix2)
            mi[ci['gy'][vi] - iy1, ci['gx'][vi] - ix1] = True
            cj = candidates[j]
            vj = (cj['gy'] >= iy1) & (cj['gy'] < iy2) & (cj['gx'] >= ix1) & (cj['gx'] < ix2)
            mj[cj['gy'][vj] - iy1, cj['gx'][vj] - ix1] = True
            inter = int(np.sum(mi & mj))
            if inter == 0:
                continue
            iou = inter / (ci['area'] + cj['area'] - inter)
            if iou > 0.3:
                if ci['area'] >= cj['area']:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    surviving = [c for c, k in zip(candidates, keep) if k]
    print(f"  合并完成，总细胞数: {len(surviving)}", flush=True)
    return surviving


# --- 大图过滤链（基于坐标，无全图 mask）---
def _filter_and_rank_tile(candidates, raw_image, settings):
    gray         = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    H_img, W_img = gray.shape

    hull_comp_thr  = settings['hull_comp']
    circle_comp_thr = settings['circle_comp']
    area_ratio     = settings['area_ratio']
    circ_thr       = settings['circularity']
    min_pixels     = settings['min_pixels']
    top_pct        = settings['brightness_top_pct']

    # Step1: 完整度过滤（bbox 近似）
    filtered = []
    for c in candidates:
        area = c['area']
        bbox = c['bbox']
        bh   = bbox[2] - bbox[0] + 1
        bw   = bbox[3] - bbox[1] + 1
        er   = max(bh, bw) / 2
        hull_approx = bh * bw * 0.785
        hull_comp   = area / hull_approx if hull_approx > 0 else 0.0
        circle_area = math.pi * er * er
        circle_comp = area / circle_area if circle_area > 0 else 0.0
        if hull_comp < hull_comp_thr or circle_comp < circle_comp_thr:
            continue
        filtered.append({**c, 'er': er})
    print(f"  [Step1] 完整度过滤: {len(filtered)} / {len(candidates)}", flush=True)

    # Step2: 面积过滤
    if filtered:
        median_area = float(np.median([c['area'] for c in filtered]))
        filtered    = [c for c in filtered if c['area'] >= median_area * area_ratio]
    print(f"  [Step2] 面积过滤: {len(filtered)}", flush=True)

    # Step3: 圆形度过滤
    result = []
    for c in filtered:
        er        = c['er']
        perimeter = 2 * math.pi * er
        circ      = 4 * math.pi * c['area'] / (perimeter ** 2) if perimeter > 0 else 0.0
        if circ >= circ_thr:
            result.append(c)
    print(f"  [Step3] 圆形度过滤: {len(result)}", flush=True)

    # 亮度计算
    cell_list = []
    for c in result:
        gy, gx  = c['gy'], c['gx']
        cy_c    = int(np.mean(gy))
        cx_c    = int(np.mean(gx))
        er      = c['er']
        ir      = max(1, int(er * 0.8))
        ry1     = max(0, cy_c - ir);  ry2 = min(H_img, cy_c + ir + 1)
        rx1     = max(0, cx_c - ir);  rx2 = min(W_img, cx_c + ir + 1)
        cell_pixels = gray[ry1:ry2, rx1:rx2].flatten()
        if len(cell_pixels) > min_pixels:
            k    = max(1, int(len(cell_pixels) * top_pct))
            peak = float(np.mean(np.partition(cell_pixels, -k)[-k:]))

            bbox     = c['bbox']
            ly0, lx0 = bbox[0], bbox[1]
            lh       = bbox[2] - ly0 + 1
            lw       = bbox[3] - lx0 + 1
            local_m  = np.zeros((lh, lw), dtype=np.uint8)
            local_m[gy - ly0, gx - lx0] = 1
            contours_local, _ = cv2.findContours(
                local_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_global = [
                (cnt + np.array([[[lx0, ly0]]])).astype(np.int32)
                for cnt in contours_local
            ]
            cell_list.append({
                "brightness": peak,
                "pos": (cx_c, cy_c),
                "gy": gy, "gx": gx,
                "er": er,
                "contours": contours_global,
            })

    cell_list.sort(key=lambda x: x["brightness"], reverse=True)
    print(f"  [Step4] 有效细胞: {len(cell_list)}", flush=True)
    return cell_list


# --- 小图过滤链（基于 cellpose mask）---
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
            cell_list.append({
                "brightness": peak, "pos": (cx, cy),
                "contours": c["contours"], "mask": c["mask"], "er": c["er"],
            })

    cell_list.sort(key=lambda x: x["brightness"], reverse=True)
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

    h, w     = raw_image.shape[:2]
    H_img, W_img = h, w
    is_large = w * h > 3000 * 3000
    top_n    = settings['top_n']

    save_dir = os.path.join(results_dir, stem)
    os.makedirs(save_dir, exist_ok=True)

    if is_large and settings['large_mode'] == 1:
        print(f"  大图模式 ({w}x{h})", flush=True)
        tile_candidates = _tile_and_merge(model, raw_image, settings)
        diag_img = raw_image.copy()
        for cand in tile_candidates:
            cy = int(np.mean(cand['gy']))
            cx = int(np.mean(cand['gx']))
            cv2.circle(diag_img, (cx, cy), 15, (0, 255, 0), -1)
        diag_path = os.path.join(save_dir, f"{stem}_raw_detections.png")
        cv2.imencode('.png', diag_img)[1].tofile(diag_path)
        print(f"  诊断图已保存: {len(tile_candidates)} 个原始检测点", flush=True)
        cell_list      = _filter_and_rank_tile(tile_candidates, raw_image, settings)
        total_detected = len(tile_candidates)
    else:
        if is_large:
            print(f"  大图整图模式 ({w}x{h})", flush=True)
        masks = model.eval(
            raw_image,
            diameter=settings['diameter'],
            channels=[0, 0],
            flow_threshold=settings['flow_threshold'],
            cellprob_threshold=settings['cellprob_threshold'],
            min_size=settings['min_size'],
            niter=settings['niter'],
            resample=bool(settings['resample']),
        )[0]
        diag_img     = raw_image.copy()
        cell_ids_raw = np.unique(masks)[1:]
        for cid in cell_ids_raw:
            ys_c, xs_c = np.where(masks == cid)
            cy = int(np.mean(ys_c))
            cx = int(np.mean(xs_c))
            cv2.circle(diag_img, (cx, cy), 15, (0, 255, 0), -1)
        diag_path = os.path.join(save_dir, f"{stem}_raw_detections.png")
        cv2.imencode('.png', diag_img)[1].tofile(diag_path)
        print(f"  诊断图已保存: {len(cell_ids_raw)} 个原始检测点", flush=True)
        cell_list      = _filter_and_rank_mask(masks, raw_image, settings)
        total_detected = len(np.unique(masks)) - 1
        is_large       = False

    thickness  = settings['contour_thickness']
    font_scale = settings['font_scale']

    # ── 标注结果图 ────────────────────────────────────────────────────────────
    res_img = raw_image.copy()
    if cell_list:
        top_cells = cell_list[:top_n]
        others    = cell_list[top_n:]

        for idx, cell in enumerate(others, start=top_n + 1):
            cx, cy = cell["pos"]
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), thickness)
            cv2.putText(res_img, str(idx), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        if is_large:
            for cell in cell_list:
                cx, cy = cell["pos"]
                ir     = max(1, int(cell["er"] * 0.8))
                gy, gx = cell["gy"], cell["gx"]
                dist   = np.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
                ring   = dist >= ir
                res_img[gy[ring], gx[ring]] = (
                    res_img[gy[ring], gx[ring]] * 0.6).astype(np.uint8)
        else:
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

    with open(os.path.join(save_dir, f"{stem}_top{top_n}.csv"), 'w', encoding='utf-8') as f:
        f.write("排名,坐标X,坐标Y,亮度值\n")
        for i, cell in enumerate(cell_list[:top_n], start=1):
            cx, cy = cell["pos"]
            f.write(f"{i},{cx},{cy},{int(cell['brightness'])}\n")

    crop_dir = os.path.join(save_dir, "crop")
    os.makedirs(crop_dir, exist_ok=True)
    for rank, cell in enumerate(cell_list[:top_n], start=1):
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

    with open(os.path.join(save_dir, f"{stem}_data.csv"), 'w', encoding='utf-8') as f:
        f.write("编号,直径(px),坐标X,坐标Y,亮度值\n")
        for i, cell in enumerate(cell_list, start=1):
            cx, cy   = cell["pos"]
            diameter = round(cell["er"] * 2, 1)
            f.write(f"{i},{diameter},{cx},{cy},{int(cell['brightness'])}\n")

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
    print(f"  diameter={settings['diameter']}  top_n={settings['top_n']}  large_mode={settings['large_mode']}", flush=True)
    print(f"  tile={settings['tile_w']}x{settings['tile_h']}  overlap={settings['tile_overlap']}", flush=True)

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

    model   = load_model()
    t_start = time.time()

    for img_path in images:
        process_image(model, img_path, results_dir, settings)

    elapsed = time.time() - t_start
    print(f"\n✅ 全部完成，共耗时 {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
