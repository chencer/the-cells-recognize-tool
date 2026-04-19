import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from cellpose import models

class CellAppCP3:
    def __init__(self, root):
        self.root = root
        self.root.title("Cellpose v3.0 - 极端亮度/大透光区检测")
        self.root.geometry("1100x900")
        
        self.raw_image = None
        self.model = None
        
        self.init_model()
        self.setup_ui()

    def init_model(self):
        """加载 CP3 引擎"""
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        use_gpu = torch.backends.mps.is_available()
        try:
            self.model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
            print(f"✅ CP3 引擎就绪 (加速: {use_gpu})")
        except Exception as e:
            messagebox.showerror("环境错误", f"请确保 cellpose==3.0.11 安装正确\n{e}")

    def setup_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="1. 导入图片", command=self.load_image, width=15).pack(side=tk.LEFT, padx=5)
        self.run_btn = tk.Button(btn_frame, text="2. 识别细胞", command=self.run_analysis, 
                                 state=tk.DISABLED, width=15, bg="#28a745")
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.root, text="就绪", fg="blue")
        self.status_label.pack()

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", width=1000, height=650)
        self.canvas.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.raw_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_image(self.raw_image)
            self.run_btn.config(state=tk.NORMAL)
            self.status_label.config(text="图片已载入", fg="green")

    def run_analysis(self):
        if self.raw_image is None: return
        self.status_label.config(text="🚀 识别中...", fg="orange")
        self.root.update()
        
        try:
            # 使用之前为你优化的针对该图片的参数
            masks, flows, styles, diams = self.model.eval(
                self.raw_image, 
                diameter=120, 
                channels=[3,0], # 针对红色
                flow_threshold=0.95, 
                cellprob_threshold=-2.0,
                min_size=200,
                resample=True
            )
            self.render_results(masks)
        except Exception as e:
            messagebox.showerror("识别失败", f"{e}")

    def render_results(self, masks):
        res_img = self.raw_image.copy()
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        cell_ids = np.unique(masks)[1:]
        
        cell_list = []
        for cid in cell_ids:
            mask = (masks == cid).astype(np.uint8)
            cell_pixels = gray[mask > 0]
            
            if len(cell_pixels) > 0:
                # --- 核心修改：计算积分亮度 (Sum) ---
                # 这代表了整个细胞区域散发出的总光能
                # 面积大且亮度高的细胞，这个值会非常显著
                total_energy = np.sum(cell_pixels)
                
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cell_list.append({"energy": total_energy, "pos": (cx, cy), "mask": mask})

        # 按总能量排序（从大到小）
        cell_list.sort(key=lambda x: x['energy'], reverse=True)

        for idx, cell in enumerate(cell_list):
            contours, _ = cv2.findContours(cell['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 默认勾边：黄色
            color = (0, 255, 255)
            thickness = 2
            
            # --- 仅特别标注 TOP 1 ---
            if idx < 1:
                color = (0, 0, 255) # 红色
                thickness = 4       # 加粗，让人眼一眼看到
                label = f"BRIGHTEST #{idx+1}"
                cv2.putText(res_img, label, (cell['pos'][0]-50, cell['pos'][1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.drawContours(res_img, contours, -1, color, thickness)

        self.display_image(res_img)
        self.status_label.config(text=f"✅ 完成！", fg="green")

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