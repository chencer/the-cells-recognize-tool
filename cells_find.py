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

try:
    import ttkbootstrap as ttk
except ImportError:
    ttk = None

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_PRIMARY   = '#0A0A14'
BG_SECONDARY = '#12121F'
BG_PANEL     = '#16162A'
BG_CANVAS    = '#080810'
ACCENT_PUR   = '#7B2FBE'
ACCENT_GREEN = '#00E676'
TEXT_PRI     = '#E8E8F0'
TEXT_SEC     = '#7A7A9E'
TEXT_DIM     = '#4A4A6A'
BORDER       = '#1E1E38'
BORDER_ACT   = '#2E2E50'
GREEN_BTN    = '#1B6B3A'
GREEN_HOV    = '#28a745'
ORANGE       = '#FFA040'
TOOLBAR_BG   = '#10101E'

if sys.platform == 'win32':
    UI_FONT   = ('Microsoft YaHei', 9, 'bold')
    MONO_FONT = ('Consolas', 9)
else:
    UI_FONT   = ('Arial', 9, 'bold')
    MONO_FONT = ('Courier New', 9)


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class CellAppCP3:
    def __init__(self, root):
        self.root = root
        self.raw_image = None
        self.model = None
        self._drag_ox = 0
        self._drag_oy = 0

        self._setup_window()
        self.init_model()
        self.setup_ui()

    # ── Window ────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.overrideredirect(True)
        self.root.configure(bg=BG_PRIMARY)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"1100x750+{(sw-1100)//2}+{(sh-750)//2}")

    def _start_drag(self, event):
        self._drag_ox = event.x_root - self.root.winfo_x()
        self._drag_oy = event.y_root - self.root.winfo_y()

    def _do_drag(self, event):
        self.root.geometry(f"+{event.x_root - self._drag_ox}+{event.y_root - self._drag_oy}")

    def _minimize(self, event=None):
        self.root.overrideredirect(False)
        self.root.iconify()
        def _restore(e):
            self.root.overrideredirect(True)
            self.root.unbind('<Map>')
        self.root.bind('<Map>', _restore)

    # ── Model ─────────────────────────────────────────────────────────────────
    def init_model(self):
        import torch
        from cellpose import models
        import ssl

        torch.serialization.add_safe_globals([models.CellposeModel])
        import torch.serialization
        torch.load = lambda *a, **kw: torch.serialization.load(*a, **kw, weights_only=False)

        ssl._create_default_https_context = ssl._create_unverified_context
        print("正在定位模型文件...")
        use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()

        try:
            model_path = get_resource_path("cyto3")
            if os.path.exists(model_path):
                self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
                print(f"✅ 成功加载本地模型: {model_path}")
            else:
                self.model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
                print("✅ 使用系统默认路径加载 cyto3")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            messagebox.showerror("初始化失败", f"错误详情: {e}")

    # ── UI ────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        self._build_titlebar()
        self._build_main_area()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=BG_SECONDARY, height=32)
        bar.pack(fill='x', side='top')
        bar.pack_propagate(False)

        # Purple icon square
        icon = tk.Canvas(bar, width=14, height=14,
                         bg=BG_SECONDARY, highlightthickness=0)
        icon.pack(side='left', padx=(12, 7), pady=9)
        icon.create_rectangle(0, 0, 14, 14, fill=ACCENT_PUR, outline='')

        tk.Label(bar, text='CellAppCP3 — Cellpose v3.0',
                 bg=BG_SECONDARY, fg=TEXT_DIM,
                 font=MONO_FONT).pack(side='left')

        # Windows controls (right-aligned)
        ctrl = tk.Frame(bar, bg=BG_SECONDARY)
        ctrl.pack(side='right', fill='y')

        def _wbtn(text, cmd, is_close=False):
            b = tk.Label(ctrl, text=text, bg=BG_SECONDARY, fg=TEXT_DIM,
                         font=('Arial', 10), width=4, height=1, cursor='hand2')
            b.pack(side='left', fill='y')
            if cmd:
                b.bind('<Button-1>', lambda e: cmd())
            hbg = '#E81123' if is_close else '#2A2A3A'
            hfg = 'white'   if is_close else TEXT_DIM
            b.bind('<Enter>', lambda e: b.config(bg=hbg, fg=hfg))
            b.bind('<Leave>', lambda e: b.config(bg=BG_SECONDARY, fg=TEXT_DIM))

        _wbtn('─', self._minimize)
        _wbtn('□', None)
        _wbtn('✕', self.root.destroy, is_close=True)

        # 1-px separator
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x', side='top')

        # Drag on title bar
        bar.bind('<ButtonPress-1>', self._start_drag)
        bar.bind('<B1-Motion>',     self._do_drag)
        icon.bind('<ButtonPress-1>', self._start_drag)
        icon.bind('<B1-Motion>',     self._do_drag)

    def _build_main_area(self):
        self.main_frame = tk.Frame(self.root, bg=BG_CANVAS)
        self.main_frame.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg=BG_CANVAS, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        self._build_floating_toolbar()

    def _on_canvas_resize(self, event):
        if self.raw_image is None:
            self._draw_empty_state()

    def _draw_empty_state(self):
        self.canvas.delete('empty_state')
        self.canvas.update_idletasks()
        cx = max(self.canvas.winfo_width() // 2, 2)
        cy = max(self.canvas.winfo_height() // 2, 2)
        s = 24
        self.canvas.create_rectangle(
            cx-s, cy-s, cx+s, cy+s,
            outline=BORDER_ACT, dash=(4, 4), width=2, tags='empty_state')
        self.canvas.create_text(
            cx, cy, text='+', fill=TEXT_DIM,
            font=('Arial', 18, 'bold'), tags='empty_state')
        self.canvas.create_text(
            cx, cy+52, text='导入荧光细胞图片开始分析',
            fill=TEXT_DIM, font=('Arial', 10), tags='empty_state')

    def _build_floating_toolbar(self):
        # Outer frame acts as 1-px border
        outer = tk.Frame(self.main_frame, bg=BORDER_ACT)
        outer.place(relx=0.5, rely=1.0, anchor='s', y=-14)

        inner = tk.Frame(outer, bg=TOOLBAR_BG)
        inner.pack(padx=1, pady=1)

        pad = tk.Frame(inner, bg=TOOLBAR_BG, padx=20, pady=8)
        pad.pack()

        # ── Import button ──────────────────────────────────────────────────
        self.import_btn = tk.Button(
            pad, text='导入图片',
            bg=BG_PANEL, fg=TEXT_PRI,
            activebackground='#252540', activeforeground=TEXT_PRI,
            font=UI_FONT, relief='flat', bd=0,
            padx=16, pady=6, cursor='hand2',
            command=self.load_image,
        )
        self.import_btn.pack(side='left', padx=(0, 6))
        self._hover(self.import_btn, BG_PANEL, '#252540')

        # ── Run button (disabled initially) ───────────────────────────────
        self.run_btn = tk.Button(
            pad, text='细胞识别',
            bg='#1A1A2E', fg=TEXT_DIM,
            activebackground='#1A1A2E', activeforeground=TEXT_DIM,
            font=UI_FONT, relief='flat', bd=0,
            padx=16, pady=6, cursor='arrow',
            state='disabled', disabledforeground=TEXT_DIM,
            command=self.run_analysis,
        )
        self.run_btn.pack(side='left')

        # ── Divider ────────────────────────────────────────────────────────
        tk.Frame(pad, bg=BORDER_ACT, width=1, height=22).pack(side='left', padx=12)

        # ── Status ─────────────────────────────────────────────────────────
        sf = tk.Frame(pad, bg=TOOLBAR_BG)
        sf.pack(side='left')

        self._dot = tk.Canvas(sf, width=7, height=7,
                              bg=TOOLBAR_BG, highlightthickness=0)
        self._dot.pack(side='left', padx=(0, 6), pady=1)
        self._dot_draw(ACCENT_PUR)

        self.status_label = tk.Label(
            sf, text='准备就绪',
            bg=TOOLBAR_BG, fg=TEXT_SEC,
            font=MONO_FONT,
        )
        self.status_label.pack(side='left')

    def _hover(self, widget, normal_bg, hover_bg):
        widget.bind('<Enter>', lambda e: widget.config(bg=hover_bg))
        widget.bind('<Leave>', lambda e: widget.config(bg=normal_bg))

    def _dot_draw(self, color):
        self._dot.delete('all')
        self._dot.create_oval(1, 1, 6, 6, fill=color, outline='')

    def _set_status(self, text, fg=TEXT_SEC, dot=ACCENT_PUR):
        self.status_label.config(text=text, fg=fg)
        self._dot_draw(dot)

    def _enable_run_btn(self):
        self.run_btn.config(
            state='normal',
            bg=GREEN_BTN, fg='white',
            activebackground=GREEN_HOV, activeforeground='white',
            cursor='hand2',
        )
        self._hover(self.run_btn, GREEN_BTN, GREEN_HOV)

    # ── Actions ───────────────────────────────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.raw_image = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_image(self.raw_image)
            self._enable_run_btn()
            self._set_status('图片已载入', TEXT_PRI, ACCENT_PUR)

    def run_analysis(self):
        if self.raw_image is None:
            return
        self._set_status('正在分析...', ORANGE, ORANGE)
        self.root.update()
        try:
            masks, flows, styles = self.model.eval(
                self.raw_image,
                diameter=120,
                channels=[0, 0],
                flow_threshold=0.95,
                cellprob_threshold=1.0,
                min_size=200,
                resample=True,
            )[:3]
            self.render_results(masks)
        except Exception as e:
            messagebox.showerror("识别失败", f"{e}")

    # ── Render ────────────────────────────────────────────────────────────────
    def render_results(self, masks):
        res_img = self.raw_image.copy()
        gray    = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        cell_ids = np.unique(masks)[1:]
        total_detected = len(cell_ids)

        # ── Step 1: completeness filter ──────────────────────────────────────
        candidates = []
        for cid in cell_ids:
            mask = (masks == cid).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  cell {cid}: no contours → skip")
                continue
            mask_area = float(np.sum(mask > 0))

            touches = (np.any(mask[0, :] > 0) or np.any(mask[-1, :] > 0) or
                       np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0))

            hull      = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            hull_comp = mask_area / hull_area if hull_area > 0 else 0.0

            (_, _), er = cv2.minEnclosingCircle(contours[0])
            circle_area = np.pi * er * er
            circle_comp = mask_area / circle_area if circle_area > 0 else 0.0

            completeness = circle_comp if touches else hull_comp
            method       = "circle"    if touches else "hull"

            print(f"  cell {cid}: mask={mask_area:.0f}px  hull_comp={hull_comp:.3f}"
                  f"  circle_comp={circle_comp:.3f}  touches={'Y' if touches else 'N'}"
                  f"  → using {method}={completeness:.3f}")

            if completeness < 0.85:
                print(f"  [skip-completeness] cell {cid}: {method}_comp={completeness:.3f} < 0.85")
                continue

            perimeter = cv2.arcLength(contours[0], True)
            candidates.append({
                "cid": cid, "mask": mask, "mask_area": mask_area,
                "contours": contours, "perimeter": perimeter,
            })
        print(f"[Step1] 完整度过滤后: {len(candidates)} / {total_detected}")

        # ── Step 2: area filter ───────────────────────────────────────────────
        if candidates:
            median_area = float(np.median([c["mask_area"] for c in candidates]))
            area_thresh = median_area * 0.1
            before2 = len(candidates)
            filtered = []
            for c in candidates:
                if c["mask_area"] < area_thresh:
                    print(f"  [skip-area] cell {c['cid']}: area={c['mask_area']:.0f} < thresh={area_thresh:.0f}")
                    continue
                filtered.append(c)
            candidates = filtered
            print(f"[Step2] 面积过滤后: {len(candidates)} / {before2}"
                  f"  (中位数={median_area:.0f}, 阈值={area_thresh:.0f})")

        # ── Step 3: circularity filter ────────────────────────────────────────
        filtered = []
        before3 = len(candidates)
        for c in candidates:
            circ = (4 * np.pi * c["mask_area"] / (c["perimeter"] ** 2)
                    if c["perimeter"] > 0 else 0.0)
            if circ < 0.4:
                print(f"  [skip-circularity] cell {c['cid']}: circularity={circ:.3f} < 0.40")
                continue
            filtered.append(c)
        candidates = filtered
        print(f"[Step3] 圆形度过滤后: {len(candidates)} / {before3}")

        # ── Brightness ────────────────────────────────────────────────────────
        cell_list = []
        for c in candidates:
            cell_pixels = gray[c["mask"] > 0]
            if len(cell_pixels) > 50:
                sorted_px = np.sort(cell_pixels)[::-1]
                top5 = max(1, int(len(sorted_px) * 0.05))
                peak  = float(np.mean(sorted_px[:top5]))
                M = cv2.moments(c["mask"])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cell_list.append({
                        "brightness": peak,
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
            bv = int(top["brightness"])
            cv2.drawContours(res_img, top["contours"], -1, (0, 255, 0), 2)
            cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(res_img, f"({cx}, {cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"最亮细胞坐标: ({cx}, {cy})  亮度值: {bv}")
            self._set_status(f'完成 · ({cx}, {cy})  亮度={bv}', ACCENT_GREEN, ACCENT_GREEN)
        else:
            self._set_status('完成（无有效细胞）', TEXT_SEC, ACCENT_PUR)

        self.display_image(res_img)

    # ── Display ───────────────────────────────────────────────────────────────
    def display_image(self, img):
        self.canvas.delete('empty_state')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        cw = self.canvas.winfo_width()  or 1100
        ch = self.canvas.winfo_height() or 718
        ratio = min(cw / w, ch / h)
        nw, nh = int(w * ratio), int(h * ratio)
        img_pil = Image.fromarray(img_rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('image')
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img,
                                  anchor='center', tags='image')


if __name__ == "__main__":
    if ttk is not None:
        root = ttk.Window(themename='darkly')
    else:
        root = tk.Tk()
    app = CellAppCP3(root)
    root.mainloop()
