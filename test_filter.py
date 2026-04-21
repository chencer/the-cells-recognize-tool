"""Headless test: verify all three filter steps without GUI."""
import cv2
import numpy as np
import os


def simulate_render(raw_image, masks, output_path):
    res_img = raw_image.copy()
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    H_img, W_img = gray.shape
    cell_ids = np.unique(masks)[1:]
    total_detected = len(cell_ids)

    # Step 1: completeness filter
    # boundary cells → minEnclosingCircle completeness (hull fails for convex caps)
    # non-boundary cells → convex hull completeness
    candidates = []
    for cid in cell_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"  cell {cid}: no contours → skip")
            continue
        mask_area = float(np.sum(mask > 0))

        touches = (np.any(mask[0, :] > 0) or np.any(mask[-1, :] > 0) or
                   np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0))

        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        hull_comp = mask_area / hull_area if hull_area > 0 else 0.0

        (_, _), er = cv2.minEnclosingCircle(contours[0])
        circle_area = np.pi * er * er
        circle_comp = mask_area / circle_area if circle_area > 0 else 0.0

        if touches:
            completeness = circle_comp
            method = "circle"
        else:
            completeness = hull_comp
            method = "hull"

        print(f"  cell {cid}: mask={mask_area:.0f}px  hull_comp={hull_comp:.3f}"
              f"  circle_comp={circle_comp:.3f}  touches={'Y' if touches else 'N'}"
              f"  → using {method}={completeness:.3f}")

        if completeness < 0.7:
            print(f"  [skip-completeness] cell {cid}: {method}_comp={completeness:.3f} < 0.70")
            continue

        perimeter = cv2.arcLength(contours[0], True)
        candidates.append({
            "cid": cid, "mask": mask, "mask_area": mask_area,
            "contours": contours, "perimeter": perimeter,
        })
    print(f"[Step1] 完整度过滤后: {len(candidates)} / {total_detected}")

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
    6 synthetic cells:
      1. full circle r=50, center (150,200), bright=200        → PASS all filters
      2. full circle r=50, touching right boundary (80% inside) → PASS (circle_comp≈0.8)
      3. partial circle at top (centroid y=15, r=50) ~30% vis   → FAIL step1 (circle_comp<0.7)
      4. crescent ring (low hull completeness ~0.42)            → FAIL step1 (hull_comp<0.7)
      5. tiny dot r=4 (area << median*0.1)                     → FAIL step2
      6. elongated thin strip 120×6 (low circularity ~0.15)    → FAIL step3
    Expected final count: 2 (cells 1 and 2)
    """
    H, W = 400, 700
    img = np.zeros((H, W, 3), dtype=np.uint8)
    masks = np.zeros((H, W), dtype=np.int32)

    def place_circle(cid, cx, cy, r, brightness):
        cv2.circle(img, (cx, cy), r, (brightness,)*3, -1)
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(m, (cx, cy), r, 1, -1)
        masks[(m > 0) & (masks == 0)] = cid

    # 1: full bright circle, not touching boundary
    place_circle(1, 150, 200, 50, 200)

    # 2: circle just barely touching right boundary (center at x=W-45, r=50 → ~5px outside)
    # ~98% inside the image → circle_comp ≈ 0.98 → should PASS
    place_circle(2, W - 45, 200, 50, 150)

    # 3: severely truncated at top — center at y=-10, r=50 → only ~40% visible
    # hull_comp ≈ 1.0 (cap is convex), circle_comp < 0.7 → should FAIL
    place_circle(3, 350, -10, 50, 180)

    # 4: crescent (outer r=55 minus inner r=45) → hull_comp ≈ 0.42 → FAIL
    cv2.circle(img, (500, 200), 55, (170,)*3, -1)
    cv2.circle(img, (500, 200), 42, (0,)*3, -1)
    m4 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m4, (500, 200), 55, 1, -1)
    cv2.circle(m4, (500, 200), 42, 0, -1)
    masks[(m4 > 0) & (masks == 0)] = 4

    # 5: tiny dot r=4 → FAIL area filter
    place_circle(5, 620, 100, 4, 180)

    # 6: thin horizontal strip 120×6 → FAIL circularity filter
    cv2.rectangle(img, (50, 340), (170, 346), (160,)*3, -1)
    m6 = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(m6, (50, 340), (170, 346), 1, -1)
    masks[(m6 > 0) & (masks == 0)] = 6

    out = os.path.join(os.path.dirname(__file__), "test_result.png")
    print("=== Per-cell completeness report ===")
    cell_list, total = simulate_render(img, masks, out)

    print(f"\n[TEST] total={total}, final={len(cell_list)}")
    assert total == 6, f"Expected 6 detected, got {total}"
    assert len(cell_list) == 2, f"Expected 2 after all filters, got {len(cell_list)}"

    # cell 2 (boundary-touching but complete) must be kept
    kept_x = [c["pos"][0] for c in cell_list]
    assert any(x > W - 100 for x in kept_x), \
        "Boundary-touching complete cell should be KEPT"

    # cell 3 (30% visible truncated) must NOT be in results
    kept_y = [c["pos"][1] for c in cell_list]
    assert not any(y < 50 for y in kept_y), \
        "Severely truncated boundary cell (y≈15) should be FILTERED"

    print("[TEST] PASS: boundary truncated filtered, complete boundary cell kept")
    print(f"Result: {out}")
    return out


if __name__ == "__main__":
    test_all_filters()
