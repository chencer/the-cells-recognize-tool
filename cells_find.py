import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from cellpose import models
import sys
import ssl

# 1. 定义资源路径获取函数 (位于类之外)
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class CellAppCP3:
    def __init__(self, root):
        self.root = root
        self.root.title("Cellpose v3.0")
        self.root.geometry("1100x900")
        
        self.raw_image = None
        self.model = None
        
        self.init_model()
        self.setup_ui()

    def init_model(self):
        import torch
        from cellpose import models
        import os
        import ssl
        
        # --- 核心修复：允许加载自定义模型权重 ---
        # 这行代码告诉 PyTorch 2.6+ 不要使用严格的 weights_only 模式
        torch.serialization.add_safe_globals([models.CellposeModel]) 
        # 如果上面那行不行，可以用下面这个最暴力的（推荐）：
        import torch.serialization
        torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)
        # ---------------------------------------

        ssl._create_default_https_context = ssl._create_unverified_context
        print("正在定位模型文件...")
        
        # 打包版建议先强制 GPU=False (CPU 运行最稳)，如果对方有环境再开启
        use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
        
        try:
            # 获取打包内部或本地的模型路径
            model_path = get_resource_path("cyto3")
            
            if os.path.exists(model_path):
                # 在 CP3 中，通过 pretrained_model 传入路径
                self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
                print(f"✅ 成功加载本地模型: {model_path}")
            else:
                # 本地调试备选
                self.model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
                print("✅ 使用系统默认路径加载 cyto3")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            messagebox.showerror("初始化失败", f"错误详情: {e}")

    def setup_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="1. 导入图片", command=self.load_image, width=15).pack(side=tk.LEFT, padx=5)
        self.run_btn = tk.Button(btn_frame, text="2. 细胞识别", command=self.run_analysis, 
                                 state=tk.DISABLED, width=15, bg="#28a745")
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.root, text="准备就绪", fg="blue")
        self.status_label.pack()

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", width=1000, height=650)
        self.canvas.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.raw_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_image(self.raw_image)
            self.run_btn.config(state=tk.NORMAL)
            self.status_label.config(text="图片载入成功", fg="green")

    def run_analysis(self):
        if self.raw_image is None: return
        self.status_label.config(text="正在分析中...", fg="orange")
        self.root.update()
        
        try:
            # --- 核心修复：去掉了 diams，并加上 [:3] 保证绝对兼容 ---
            masks, flows, styles = self.model.eval(
                self.raw_image, 
                diameter=120, 
                channels=[0,0], 
                flow_threshold=0.95, 
                cellprob_threshold=1.0,
                min_size=200,
                resample=True
            )[:3] 
            # ----------------------------------------------------
            
            self.render_results(masks)
        except Exception as e:
            messagebox.showerror("识别失败", f"{e}")

    def render_results(self, masks):
        res_img = self.raw_image.copy()
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        cell_ids = np.unique(masks)[1:]
        total_detected = len(cell_ids)

        # ── Step 1: 凸包完整度过滤 (mask像素数 / 凸包面积 >= 70%) ──────────────
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
                print(f"  [skip-completeness] cell {cid}: completeness={completeness:.3f} < 0.70")
                continue
            perimeter = cv2.arcLength(contours[0], True)
            candidates.append({
                "cid": cid, "mask": mask, "mask_area": mask_area,
                "contours": contours, "perimeter": perimeter,
            })
        print(f"[Step1] 凸包完整度过滤后: {len(candidates)} / {total_detected}")

        # ── Step 2: 面积过滤 (< 中位数面积 × 10% 剔除) ───────────────────────
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

        # ── Step 3: 圆形度过滤 (4π×面积/周长² >= 0.4) ───────────────────────
        filtered = []
        for c in candidates:
            circ = (4 * np.pi * c["mask_area"] / (c["perimeter"] ** 2)
                    if c["perimeter"] > 0 else 0.0)
            if circ < 0.4:
                print(f"  [skip-circularity] cell {c['cid']}: circularity={circ:.3f} < 0.40")
                continue
            filtered.append(c)
        print(f"[Step3] 圆形度过滤后: {len(filtered)} / {len(candidates)}")
        candidates = filtered

        # ── 计算亮度，构建最终列表 ──────────────────────────────────────────
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
            # 普通细胞：黄色轮廓描边，无中心点
            for cell in cell_list[1:]:
                cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), 2)

            # 最亮细胞：绿色轮廓描边 + 绿色中心点 + 绿色坐标
            top = cell_list[0]
            cx, cy = top["pos"]
            brightness_val = int(top["brightness"])
            cv2.drawContours(res_img, top["contours"], -1, (0, 255, 0), 2)
            cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(res_img, f"({cx}, {cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status_msg = f"✅ 完成  最亮细胞: 坐标({cx}, {cy})  亮度={brightness_val}"
            print(f"最亮细胞坐标: ({cx}, {cy})  亮度值: {brightness_val}")
        else:
            status_msg = "✅ 完成（未检测到有效细胞）"

        self.display_image(res_img)
        self.status_label.config(text=status_msg, fg="green")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio = min(1000/w, 650/h)
        new_size = (int(w*ratio), int(h*ratio))
        img_pil = Image.fromarray(img_rgb).resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(500, 325, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = CellAppCP3(root)
    root.mainloop()
