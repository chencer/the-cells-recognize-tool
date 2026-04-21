"""Headless test: verify all three filter steps without GUI."""
import cv2
import numpy as np
import os


def simulate_render(raw_image, masks, output_path):
    res_img = raw_image.copy()
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    cell_ids = np.unique(masks)[1:]
    total_detected = len(cell_ids)

    # Step 1: convex hull completeness
    candidates = []
    for cid in cell_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        mask_area = float(np.sum(mask > 0))
        completeness = mask_area / hull_area
        if completeness < 0.7:
            print(f"  [skip-completeness] cell {cid}: completeness={completeness:.3f}")
            continue
        perimeter = cv2.arcLength(contours[0], True)
        candidates.append({
            "cid": cid, "mask": mask, "mask_area": mask_area,
            "contours": contours, "perimeter": perimeter,
        })
    print(f"[Step1] 凸包完整度过滤后: {len(candidates)} / {total_detected}")

    # Step 2: area filter
    if candidates:
        median_area = float(np.median([c["mask_area"] for c in candidates]))
        area_thresh = median_area * 0.1
        filtered = []
        for c in candidates:
            if c["mask_area"] < area_thresh:
                print(f"  [skip-area] cell {c['cid']}: area={c['mask_area']:.0f} < thresh={area_thresh:.0f}")
                continue
            filtered.append(c)
        print(f"[Step2] 面积过滤后: {len(filtered)} / {len(candidates)}  "
              f"(中位数={median_area:.0f}, 阈值={area_thresh:.0f})")
        candidates = filtered

    # Step 3: circularity filter
    filtered = []
    for c in candidates:
        circ = (4 * np.pi * c["mask_area"] / (c["perimeter"] ** 2)
                if c["perimeter"] > 0 else 0.0)
        if circ < 0.4:
            print(f"  [skip-circularity] cell {c['cid']}: circularity={circ:.3f}")
            continue
        filtered.append(c)
    print(f"[Step3] 圆形度过滤后: {len(filtered)} / {len(candidates)}")
    candidates = filtered

    cell_list = []
    for c in candidates:
        cell_pixels = gray[c["mask"] > 0]
        if len(cell_pixels) > 50:
            sorted_px = np.sort(cell_pixels)[::-1]
            top5 = max(1, int(len(sorted_px) * 0.05))
            peak_brightness = float(np.mean(sorted_px[:top5]))
            M = cv2.moments(c["mask"])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cell_list.append({
                    "brightness": peak_brightness,
                    "pos": (cx, cy),
                    "contours": c["contours"],
                })

    print(f"过滤前细胞数: {total_detected}  最终有效细胞: {len(cell_list)}")
    cell_list.sort(key=lambda x: x["brightness"], reverse=True)

    if cell_list:
        for cell in cell_list[1:]:
            cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), 2)
        top = cell_list[0]
        cx, cy = top["pos"]
        cv2.drawContours(res_img, top["contours"], -1, (0, 255, 0), 2)
        cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(res_img, f"({cx}, {cy})", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"最亮细胞坐标: ({cx}, {cy})  亮度值: {int(top['brightness'])}")

    cv2.imwrite(output_path, res_img)
    return cell_list, total_detected


def test_all_filters():
    """
    5 synthetic cells:
      1. full circle r=50, center (150,200), bright=200     → PASS all
      2. full circle r=48, touching right boundary           → PASS (completeness high, complete cell)
      3. crescent ring (low completeness ~0.44)              → FAIL step1
      4. tiny dot r=4 (area << median*0.1)                   → FAIL step2
      5. elongated thin strip (low circularity ~0.15)        → FAIL step3
    Expected final count: 2
    """
    H, W = 400, 700
    img = np.zeros((H, W, 3), dtype=np.uint8)
    masks = np.zeros((H, W), dtype=np.int32)

    def place_circle(cid, cx, cy, r, brightness):
        cv2.circle(img, (cx, cy), r, (brightness,)*3, -1)
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(m, (cx, cy), r, 1, -1)
        masks[(m > 0) & (masks == 0)] = cid

    # 1: full bright circle
    place_circle(1, 150, 200, 50, 200)
    # 2: full circle touching right edge (center at W-10, r=50 → right at W+40)
    place_circle(2, W - 10, 200, 50, 150)
    # 3: crescent (outer r=55 minus inner r=45)
    cv2.circle(img, (400, 200), 55, (170,)*3, -1)
    cv2.circle(img, (400, 200), 42, (0,)*3, -1)
    m3 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m3, (400, 200), 55, 1, -1)
    cv2.circle(m3, (400, 200), 42, 0, -1)
    masks[(m3 > 0) & (masks == 0)] = 3
    # 4: tiny dot r=4
    place_circle(4, 550, 100, 4, 180)
    # 5: thin horizontal strip 120×6 px
    cv2.rectangle(img, (50, 320), (170, 326), (160,)*3, -1)
    m5 = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(m5, (50, 320), (170, 326), 1, -1)
    masks[(m5 > 0) & (masks == 0)] = 5

    out = os.path.join(os.path.dirname(__file__), "test_result.png")
    cell_list, total = simulate_render(img, masks, out)

    print(f"\n[TEST] total={total}, final={len(cell_list)}")
    assert total == 5, f"Expected 5 detected, got {total}"
    assert len(cell_list) == 2, f"Expected 2 after all filters, got {len(cell_list)}"
    # boundary-touching complete cell must be kept
    kept_x = [c["pos"][0] for c in cell_list]
    assert any(x > W - 100 for x in kept_x), "Boundary-touching complete cell should be kept"
    print("[TEST] PASS: all three filters work correctly")
    print(f"Result saved to: {out}")
    return out


if __name__ == "__main__":
    print("=== Full filter test ===")
    test_all_filters()
