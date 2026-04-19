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
                channels=[3,0], 
                flow_threshold=0.95, 
                cellprob_threshold=-2.0,
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
        
        cell_list = []
        for cid in cell_ids:
            mask = (masks == cid).astype(np.uint8)
            cell_pixels = gray[mask > 0]
            if len(cell_pixels) > 0:
                total_energy = np.sum(cell_pixels)
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cell_list.append({"energy": total_energy, "pos": (cx, cy), "mask": mask})

        cell_list.sort(key=lambda x: x['energy'], reverse=True)

        for idx, cell in enumerate(cell_list):
            contours, _ = cv2.findContours(cell['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 0, 255) if idx < 2 else (0, 255, 255) # 前2红，其余黄
            thickness = 4 if idx < 2 else 2
            
            if idx < 2:
                cv2.putText(res_img, f"BRIGHTEST #{idx+1}", (cell['pos'][0]-50, cell['pos'][1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.drawContours(res_img, contours, -1, color, thickness)

        self.display_image(res_img)
        self.status_label.config(text=f"✅ 完成 已标注2个目标", fg="green")

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
