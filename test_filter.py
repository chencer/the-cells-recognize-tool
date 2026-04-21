"""Headless test: verify edge filter and annotation logic without GUI."""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def simulate_render(raw_image, masks, output_path):
    res_img = raw_image.copy()
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cell_ids = np.unique(masks)[1:]

    total_detected = len(cell_ids)
    skipped_incomplete = 0
    cell_list = []

    for cid in cell_ids:
        mask = (masks == cid).astype(np.uint8)

        # Convex hull completeness filter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"  [skip] cell {cid}: no contours")
            skipped_incomplete += 1
            continue
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            print(f"  [skip] cell {cid}: hull_area=0")
            skipped_incomplete += 1
            continue
        mask_area = float(np.sum(mask > 0))
        completeness = mask_area / hull_area
        if completeness < 0.7:
            print(f"  [skip] cell {cid}: completeness={completeness:.3f} < 0.70")
            skipped_incomplete += 1
            continue

        cell_pixels = gray[mask > 0]
        if len(cell_pixels) > 50:
            sorted_pixels = np.sort(cell_pixels)[::-1]
            top_5_count = max(1, int(len(sorted_pixels) * 0.05))
            peak_brightness = np.mean(sorted_pixels[:top_5_count])

            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cell_list.append({
                    "brightness": peak_brightness,
                    "pos": (cx, cy),
                    "contours": contours,
                })

    print(f"过滤前细胞数: {total_detected}  过滤后细胞数: {len(cell_list)}  (过滤掉: {skipped_incomplete})")

    cell_list.sort(key=lambda x: x['brightness'], reverse=True)

    if cell_list:
        for cell in cell_list[1:]:
            cv2.drawContours(res_img, cell['contours'], -1, (0, 255, 255), 2)

        top = cell_list[0]
        cx, cy = top['pos']
        brightness_val = int(top['brightness'])
        cv2.drawContours(res_img, top['contours'], -1, (0, 255, 0), 2)
        cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(res_img, f"({cx}, {cy})", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"最亮细胞坐标: ({cx}, {cy})  亮度值: {brightness_val}")
    else:
        print("未检测到有效细胞")

    cv2.imwrite(output_path, res_img)
    print(f"结果图已保存: {output_path}")
    return cell_list, total_detected, skipped_incomplete


def test_synthetic():
    """Test completeness filter:
    - Cell 1: full circle (completeness ~1.0) → keep
    - Cell 2: full circle touching boundary (completeness ~1.0) → keep (not penalized for touching edge)
    - Cell 3: crescent/C-shape with large hull but small mask (completeness ~0.4) → filter
    """
    H, W = 400, 600
    img = np.zeros((H, W, 3), dtype=np.uint8)
    masks = np.zeros((H, W), dtype=np.int32)

    # Cell 1: full circle, center (150, 200), radius 50, bright
    cv2.circle(img, (150, 200), 50, (200, 200, 200), -1)
    mask1 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask1, (150, 200), 50, 1, -1)
    masks[mask1 > 0] = 1

    # Cell 2: full circle touching right boundary (center at x=555, r=50 → right edge at x=605 > W=600)
    cv2.circle(img, (555, 200), 50, (150, 150, 150), -1)
    mask2 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask2, (555, 200), 50, 1, -1)
    masks[(mask2 > 0) & (masks == 0)] = 2

    # Cell 3: crescent shape — large outer circle minus inner circle → low completeness
    # Outer circle r=60, inner hole r=50 → ring area ≈ π(60²-50²) ≈ 3456px
    # Hull ≈ π*60² ≈ 11310 → completeness ≈ 0.31 → should be filtered
    mask3 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask3, (400, 300), 60, 1, -1)
    cv2.circle(mask3, (400, 300), 45, 0, -1)   # cut out center → crescent
    cv2.circle(img, (400, 300), 60, (170, 170, 170), -1)
    cv2.circle(img, (400, 300), 45, (0, 0, 0), -1)
    masks[(mask3 > 0) & (masks == 0)] = 3

    output_path = os.path.join(os.path.dirname(__file__), "test_output_synthetic.png")
    cell_list, total, skipped = simulate_render(img, masks, output_path)

    print(f"\n[TEST] Total detected: {total}, After filter: {len(cell_list)}, Skipped: {skipped}")
    assert total == 3, f"Expected 3 detected, got {total}"
    # Cell 2 (touching boundary) must be kept — completeness is high
    kept_positions = [c['pos'] for c in cell_list]
    assert any(x > 500 for x, y in kept_positions), \
        "Cell touching boundary should be KEPT (completeness >= 0.7)"
    # Cell 3 (crescent) must be filtered — completeness is low
    assert skipped >= 1, "Crescent cell should be filtered (completeness < 0.7)"
    print("[TEST] PASS: boundary-touching complete cell kept, crescent cell filtered")
    return output_path


if __name__ == "__main__":
    print("=== Synthetic test ===")
    out = test_synthetic()
    print(f"\nSynthetic output: {out}")
