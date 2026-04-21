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

        # Edge boundary filter
        if (np.any(mask[0, :] > 0) or np.any(mask[-1, :] > 0) or
                np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0)):
            skipped_incomplete += 1
            continue

        # Convex hull completeness filter
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
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                _, er = cv2.minEnclosingCircle(contours[0])
                cell_list.append({
                    "brightness": peak_brightness,
                    "pos": (cx, cy),
                    "radius": int(er),
                })

    print(f"过滤前细胞数: {total_detected}  过滤后细胞数: {len(cell_list)}  (过滤掉: {skipped_incomplete})")

    cell_list.sort(key=lambda x: x['brightness'], reverse=True)

    if cell_list:
        for cell in cell_list[1:]:
            cv2.circle(res_img, cell['pos'], cell['radius'], (0, 255, 255), 2)

        top = cell_list[0]
        cx, cy = top['pos']
        brightness_val = int(top['brightness'])
        cv2.circle(res_img, (cx, cy), top['radius'], (0, 255, 0), 2)
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
    """Create synthetic image with one edge cell and two full cells."""
    H, W = 400, 600
    img = np.zeros((H, W, 3), dtype=np.uint8)
    masks = np.zeros((H, W), dtype=np.int32)

    # Cell 1: full circle, center (150, 200), radius 50, bright
    cv2.circle(img, (150, 200), 50, (200, 200, 200), -1)
    mask1 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask1, (150, 200), 50, 1, -1)
    masks[mask1 > 0] = 1

    # Cell 2: full circle, center (400, 200), radius 45, medium
    cv2.circle(img, (400, 200), 45, (150, 150, 150), -1)
    mask2 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask2, (400, 200), 45, 1, -1)
    masks[mask2 > 0] = 2

    # Cell 3 (EDGE): circle centered at (5, 300), mostly outside image — only ~30% visible
    cv2.circle(img, (5, 300), 50, (180, 180, 180), -1)
    mask3 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask3, (5, 300), 50, 1, -1)
    masks[(mask3 > 0) & (masks == 0)] = 3

    output_path = os.path.join(os.path.dirname(__file__), "test_output_synthetic.png")
    cell_list, total, skipped = simulate_render(img, masks, output_path)

    print(f"\n[TEST] Total detected: {total}, After filter: {len(cell_list)}, Skipped: {skipped}")
    assert len(cell_list) == 2, f"Expected 2 valid cells, got {len(cell_list)}"
    assert total == 3, f"Expected 3 detected, got {total}"
    assert skipped >= 1, "Edge cell should have been filtered"
    print("[TEST] PASS: edge cell correctly filtered out")
    return output_path


if __name__ == "__main__":
    print("=== Synthetic test ===")
    out = test_synthetic()
    print(f"\nSynthetic output: {out}")
