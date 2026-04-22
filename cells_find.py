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
import math
import threading
import queue

try:
    import ttkbootstrap as ttk
except ImportError:
    ttk = None

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_WIN        = '#020203'
BG_TITLEBAR   = '#08010F'
TOOLBAR_BG    = '#120920'
PANEL_BG      = '#0D0718'
PANEL_HDR_BG  = '#0A0520'
BORDER        = '#2D1550'
ACCENT_PUR    = '#7C3AED'
ACCENT_PUR_H  = '#9333EA'
ACCENT_GREEN  = '#00E676'
ACCENT_GREEN2 = '#00c766'
ACCENT_YELLOW = '#FFD600'
TEXT_PRI      = '#E0E0E0'
TEXT_SEC      = '#9E9E9E'
TEXT_DIM      = '#4A2A6A'
GLOW_BORDER   = '#7C3AED'

PANEL_W      = 280  # fixed right-panel pixel width
PANEL_FONT   = ('Courier New', 10)

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


def _smooth_pill(canvas, x1, y1, x2, y2, r, fill, outline, tag=''):
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    pts = [
        x1+r, y1,   x2-r, y1,
        x2,   y1,   x2,   y1+r,
        x2,   y2-r, x2,   y2,
        x2-r, y2,   x1+r, y2,
        x1,   y2,   x1,   y2-r,
        x1,   y1+r, x1,   y1,
    ]
    kw = {'smooth': True, 'fill': fill, 'outline': outline}
    if tag:
        kw['tags'] = tag
    canvas.create_polygon(pts, **kw)



class CellAppCP3:
    def __init__(self, root):
        self.root = root
        self.raw_image = None
        self.model = None
        self._drag_ox = 0
        self._drag_oy = 0

        # Animation state
        self._pulse_running = False
        self._pulse_t       = 0.0
        self._spin_running  = False
        self._spin_angle    = 0
        self._spin_after_id = None

        # Background gradient cache
        self._bg_size = (0, 0)
        self._bg_img  = None

        self._result_queue = queue.Queue()
        self._model_queue  = queue.Queue()

        self.result_img      = None
        self._showing_result = True
        self.image_path      = None
        self.cell_list       = None

        self._setup_window()
        self.setup_ui()
        self.root.after(120, self._start_model_load)

    # ── Window ────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.overrideredirect(True)
        self.root.configure(bg=BG_WIN)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"1400x750+{(sw-1400)//2}+{(sh-750)//2}")

    def _start_drag(self, event):
        self._drag_ox = event.x_root - self.root.winfo_x()
        self._drag_oy = event.y_root - self.root.winfo_y()

    def _do_drag(self, event):
        self.root.geometry(
            f"+{event.x_root - self._drag_ox}+{event.y_root - self._drag_oy}")

    def _minimize(self, event=None):
        self.root.overrideredirect(False)
        self.root.iconify()
        def _restore(e):
            self.root.overrideredirect(True)
            self.root.unbind('<Map>')
        self.root.bind('<Map>', _restore)

    # ── Model loading ─────────────────────────────────────────────────────────
    def _start_model_load(self):
        self._set_status('正在加载模型...', ACCENT_YELLOW, ACCENT_YELLOW)
        self._start_pulse()
        self._build_spinner_card(
            text='正在加载 Cellpose v3.0 模型',
            subtext='首次启动需要几秒，请稍候')
        t = threading.Thread(target=self._load_model_worker, daemon=True)
        t.start()
        self._check_model_result()

    def _load_model_worker(self):
        try:
            import torch
            from cellpose import models as cp_models
            import ssl as _ssl

            torch.serialization.add_safe_globals([cp_models.CellposeModel])
            import torch.serialization
            torch.load = lambda *a, **kw: torch.serialization.load(
                *a, **kw, weights_only=False)

            _ssl._create_default_https_context = _ssl._create_unverified_context
            print("正在定位模型文件...")
            use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()

            model_path = get_resource_path("cyto3")
            if os.path.exists(model_path):
                m = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
                print(f"✅ 成功加载本地模型: {model_path}")
            else:
                m = cp_models.Cellpose(gpu=use_gpu, model_type="cyto3")
                print("✅ 使用系统默认路径加载 cyto3")

            self._model_queue.put(('ok', m))
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            self._model_queue.put(('err', e))

    def _check_model_result(self):
        try:
            status, payload = self._model_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_model_result)
            return
        self._stop_pulse()
        self._destroy_spinner_card()
        if status == 'ok':
            self.model = payload
            self._set_status('准备就绪', TEXT_SEC, TEXT_DIM)
            self.import_btn.config(
                state='normal', bg=ACCENT_PUR, fg='white',
                activebackground=ACCENT_PUR_H, cursor='hand2')
            self.import_btn.bind('<Enter>',
                lambda e: self.import_btn.config(bg=ACCENT_PUR_H))
            self.import_btn.bind('<Leave>',
                lambda e: self.import_btn.config(bg=ACCENT_PUR))
            self.root.after(10, self._update_toolbar_pill)
        else:
            self._set_status('模型加载失败', '#ff5555', '#ff5555')
            messagebox.showerror("初始化失败", f"错误详情: {payload}")

    # ── UI ────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        self._build_titlebar()
        self._build_main_area()
        self.root.after(80, self._update_toolbar_pill)
        self.root.after(80, self._update_status_pill)

    # ── Title bar ─────────────────────────────────────────────────────────────
    def _build_titlebar(self):
        bar = tk.Frame(self.root, height=32, bg=BG_TITLEBAR)
        bar.pack(fill='x', side='top')
        bar.pack_propagate(False)

        tk.Label(bar, text='CellAppCP3 — Cellpose v3.0',
                 bg=BG_TITLEBAR, fg=TEXT_SEC,
                 font=MONO_FONT).place(x=12, rely=0.5, anchor='w')

        def _winbtn(parent, text, hover, cmd):
            b = tk.Label(parent, text=text, bg=BG_TITLEBAR, fg=TEXT_SEC,
                         font=('Arial', 11), width=3, cursor='hand2')
            b.pack(side='left')
            b.bind('<Enter>',    lambda e: b.config(bg=hover))
            b.bind('<Leave>',    lambda e: b.config(bg=BG_TITLEBAR))
            b.bind('<Button-1>', lambda e: cmd())
            return b

        btn_row = tk.Frame(bar, bg=BG_TITLEBAR)
        btn_row.place(relx=1.0, rely=0.5, anchor='e')

        _winbtn(btn_row, '─', '#1A0A30', self._minimize)
        _winbtn(btn_row, '✕', '#c42b1c', self.root.destroy)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x', side='top')

        bar.bind('<ButtonPress-1>', self._start_drag)
        bar.bind('<B1-Motion>',     self._do_drag)

    # ── Main area: left canvas + right panel ──────────────────────────────────
    def _build_main_area(self):
        self.main_frame = tk.Frame(self.root, bg=BG_WIN)
        self.main_frame.pack(fill='both', expand=True)

        # Right panel (fixed width, packed first so canvas fills remainder)
        tk.Frame(self.main_frame, bg=BORDER, width=1).pack(side='right', fill='y')
        self.right_frame = tk.Frame(
            self.main_frame, bg=PANEL_BG, width=PANEL_W)
        self.right_frame.pack(side='right', fill='y')
        self.right_frame.pack_propagate(False)

        # Left canvas area
        self.left_frame = tk.Frame(self.main_frame, bg=BG_WIN)
        self.left_frame.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(self.left_frame, bg=BG_WIN, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self._build_floating_toolbar()
        self._build_status_pill()
        self._build_data_panel()

    def _on_canvas_configure(self, event):
        self._update_bg_gradient()
        if self.raw_image is None:
            self._draw_empty_state()

    # ── Background gradient ───────────────────────────────────────────────────
    def _update_bg_gradient(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2 or (w, h) == self._bg_size:
            return
        self._bg_size = (w, h)

        cx, cy = w / 2, h / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.clip(
            np.sqrt((xx - cx)**2 + (yy - cy)**2) / (max(w, h) * 0.75),
            0, 1)

        r = (0x0A * (1 - dist)).astype(np.uint8)
        g = (0x03 * (1 - dist)).astype(np.uint8)
        b = (0x10 * (1 - dist)).astype(np.uint8)

        self._bg_img = ImageTk.PhotoImage(
            Image.fromarray(np.stack([r, g, b], axis=-1), 'RGB'))
        self.canvas.delete('bg_gradient')
        self.canvas.create_image(0, 0, image=self._bg_img, anchor='nw',
                                 tags='bg_gradient')
        self.canvas.tag_lower('bg_gradient')

    # ── Empty state ───────────────────────────────────────────────────────────
    def _draw_empty_state(self):
        self.canvas.delete('empty_state')
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 4:
            return
        cx, cy = w // 2, h // 2
        bw, bh = 400, 220

        self.canvas.create_rectangle(
            cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2,
            outline='#2D1550', dash=(6, 4), width=2, fill='',
            tags='empty_state')

        ix, iy = cx, cy - 28
        self.canvas.create_rectangle(
            ix-22, iy-14, ix+22, iy+14,
            outline=TEXT_DIM, width=1, fill='', tags='empty_state')
        self.canvas.create_oval(
            ix-7, iy-7, ix+7, iy+7,
            outline=TEXT_DIM, width=1, fill='', tags='empty_state')
        self.canvas.create_rectangle(
            ix-10, iy-19, ix-3, iy-14,
            outline=TEXT_DIM, width=1, fill='', tags='empty_state')
        self.canvas.create_line(
            ix-22, iy+4, ix-10, iy-6, fill=TEXT_DIM, width=1,
            tags='empty_state')

        self.canvas.create_text(
            cx, cy + 14, text='点击 "1. 导入图片" 开始',
            fill=TEXT_DIM, font=('Arial', 12), tags='empty_state')
        self.canvas.create_text(
            cx, cy + 38, text='.tif · .tiff · .png · 2048 × 1080',
            fill='#3e3e4a', font=MONO_FONT, tags='empty_state')

    # ── Floating toolbar pill (top-left of left_frame) ────────────────────────
    def _build_floating_toolbar(self):
        self._tb_pill = tk.Canvas(self.left_frame, bg=BG_WIN, highlightthickness=0)
        self._tb_pill.place(x=16, y=16)

        self._tb_frame = tk.Frame(self._tb_pill, bg=TOOLBAR_BG, padx=6, pady=6)

        self.import_btn = self._make_btn(
            self._tb_frame, '  导入图片  ',
            '#1A0A30', '#1A0A30', TEXT_DIM,
            command=self.load_image, state='disabled')
        self.import_btn.pack(side='left', padx=(0, 4))

        self.run_btn = self._make_btn(
            self._tb_frame, '  细胞识别  ',
            '#1A0A30', '#1A0A30', TEXT_DIM,
            command=self.run_analysis, state='disabled')
        self.run_btn.pack(side='left')

        self.toggle_btn = self._make_btn(
            self._tb_frame, '  原图  ',
            '#1A0A30', '#2D1050', TEXT_SEC,
            command=self._toggle_view)

        self.save_btn = self._make_btn(
            self._tb_frame, '  保存  ',
            '#0A1A0A', '#1A3020', '#00E676',
            command=self.save_results)

    def _update_toolbar_pill(self):
        self._tb_frame.update_idletasks()
        fw = self._tb_frame.winfo_reqwidth()
        fh = self._tb_frame.winfo_reqheight()
        pad = 5
        pw, ph = fw + 2*pad, fh + 2*pad

        self._tb_pill.config(width=pw, height=ph)
        self._tb_pill.delete('all')
        _smooth_pill(self._tb_pill, 0, 0, pw-1, ph-1, r=12,
                     fill='', outline='#3D1A70')
        _smooth_pill(self._tb_pill, 2, 2, pw-3, ph-3, r=10,
                     fill=TOOLBAR_BG, outline=GLOW_BORDER)
        self._tb_pill.create_window(
            pw // 2, ph // 2, window=self._tb_frame, anchor='center')

    # ── Status pill (bottom center of left_frame) ─────────────────────────────
    def _build_status_pill(self):
        self._sb_pill = tk.Canvas(self.left_frame, bg=BG_WIN, highlightthickness=0)
        self._sb_pill.place(relx=0.5, rely=1.0, anchor='s', y=-20)

        self._sb_frame = tk.Frame(self._sb_pill, bg=TOOLBAR_BG, padx=14, pady=7)

        self._dot = tk.Canvas(self._sb_frame, width=7, height=7,
                              bg=TOOLBAR_BG, highlightthickness=0)
        self._dot.pack(side='left', padx=(0, 8))
        self._dot_draw(TEXT_DIM)

        self.status_label = tk.Label(
            self._sb_frame, text='准备就绪',
            bg=TOOLBAR_BG, fg=TEXT_SEC, font=MONO_FONT)
        self.status_label.pack(side='left')

    def _update_status_pill(self):
        self._sb_frame.update_idletasks()
        fw = self._sb_frame.winfo_reqwidth()
        fh = self._sb_frame.winfo_reqheight()
        pad = 5
        pw = fw + 2*pad
        ph = fh + 2*pad
        r  = ph // 2

        self._sb_pill.config(width=pw, height=ph)
        self._sb_pill.delete('all')
        _smooth_pill(self._sb_pill, 0, 0, pw-1, ph-1, r=r,
                     fill='', outline='#3D1A70')
        _smooth_pill(self._sb_pill, 2, 2, pw-3, ph-3, r=r,
                     fill=TOOLBAR_BG, outline=GLOW_BORDER)
        self._sb_pill.create_window(
            pw // 2, ph // 2, window=self._sb_frame, anchor='center')

    # ── Button helpers ────────────────────────────────────────────────────────
    def _make_btn(self, parent, text, bg, hover_bg, fg,
                  command=None, state='normal'):
        b = tk.Button(
            parent, text=text, bg=bg, fg=fg,
            activebackground=hover_bg, activeforeground=fg,
            relief='flat', bd=0, padx=10, pady=6,
            font=UI_FONT,
            cursor='hand2' if state == 'normal' else 'arrow',
            state=state, disabledforeground=TEXT_DIM,
            command=command,
        )
        if state == 'normal':
            b.bind('<Enter>', lambda e, h=hover_bg: b.config(bg=h))
            b.bind('<Leave>', lambda e, n=bg:       b.config(bg=n))
        return b

    def _dot_draw(self, color):
        self._dot.delete('all')
        self._dot.create_oval(0, 0, 7, 7, fill=color, outline='')

    def _set_status(self, text, fg=TEXT_SEC, dot=TEXT_DIM):
        self.status_label.config(text=text, fg=fg)
        self._dot_draw(dot)
        self.root.after(10, self._update_status_pill)

    def _enable_run_btn(self):
        self.run_btn.config(
            state='normal',
            bg=ACCENT_GREEN, fg='#062512',
            activebackground=ACCENT_GREEN2, activeforeground='#062512',
            cursor='hand2',
        )
        self.run_btn.bind('<Enter>', lambda e: self.run_btn.config(bg=ACCENT_GREEN2))
        self.run_btn.bind('<Leave>', lambda e: self.run_btn.config(bg=ACCENT_GREEN))
        self.root.after(10, self._update_toolbar_pill)

    def _show_toggle_btn(self):
        self._showing_result = True
        self.toggle_btn.config(text='  原图  ')
        self.toggle_btn.pack(side='left', padx=(4, 0))
        self.save_btn.pack(side='left', padx=(4, 0))
        self.root.after(10, self._update_toolbar_pill)

    def _hide_toggle_btn(self):
        self.toggle_btn.pack_forget()
        self.save_btn.pack_forget()
        self.root.after(10, self._update_toolbar_pill)

    def _hide_save_btn(self):
        self.save_btn.pack_forget()
        self.root.after(10, self._update_toolbar_pill)

    def _toggle_view(self):
        if self._showing_result:
            self.display_image(self.raw_image)
            self.toggle_btn.config(text='  结果  ')
            self._showing_result = False
        else:
            self.display_image(self.result_img)
            self.toggle_btn.config(text='  原图  ')
            self._showing_result = True

    # ── Animation: scan sweep (removed) ──────────────────────────────────────
    def _start_scan(self): pass
    def _stop_scan(self):  pass

    # ── Animation: pulse dot ──────────────────────────────────────────────────
    def _start_pulse(self):
        self._pulse_running = True
        self._pulse_t = 0.0
        self._tick_pulse()

    def _tick_pulse(self):
        if not self._pulse_running:
            return
        self._pulse_t += 0.13
        a = 0.55 + 0.45 * math.cos(self._pulse_t)
        r = 0xFF
        g = int(0xD6 * a + 0x60 * (1 - a))
        color = f'#{r:02x}{g:02x}00'
        self._dot.delete('all')
        self._dot.create_oval(0, 0, 7, 7, fill=color, outline='')
        self.root.after(40, self._tick_pulse)

    def _stop_pulse(self):
        self._pulse_running = False

    # ── Animation: spinner card ───────────────────────────────────────────────
    def _build_spinner_card(self, text='Cellpose v3.0 分析中',
                            subtext='model: cyto3'):
        cw = self.canvas.winfo_width()  or 820
        ch = self.canvas.winfo_height() or 714

        card_w, card_h = 300, 64
        self._spin_cv = tk.Canvas(
            self.left_frame,
            width=card_w, height=card_h,
            bg=BG_WIN, highlightthickness=0)
        self._spin_cv.place(
            x=cw // 2 - card_w // 2,
            y=ch // 2 - card_h // 2)

        _smooth_pill(self._spin_cv, 0, 0, card_w-1, card_h-1, r=8,
                     fill='#08010F', outline=GLOW_BORDER)

        sx, sy, sr = 32, 32, 14
        self._spin_cv.create_oval(
            sx-sr, sy-sr, sx+sr, sy+sr,
            outline='#1e1e38', width=3, fill='', tags='sp_track')
        self._spin_cv.create_arc(
            sx-sr, sy-sr, sx+sr, sy+sr,
            start=0, extent=70, outline=ACCENT_PUR,
            width=4, style='arc', tags='sp_arc')

        self._spin_text1 = self._spin_cv.create_text(
            56, 26, anchor='w', text=text,
            fill=TEXT_PRI, font=('Arial', 10, 'bold'))
        self._spin_text2 = self._spin_cv.create_text(
            57, 44, anchor='w', text=subtext,
            fill=TEXT_DIM, font=MONO_FONT)

        self._spin_running = True
        self._spin_angle   = 0
        self._tick_spinner()

    def _set_spinner_text(self, text, subtext=''):
        if hasattr(self, '_spin_cv') and self._spin_cv.winfo_exists():
            if hasattr(self, '_spin_text1'):
                self._spin_cv.itemconfig(self._spin_text1, text=text)
            if hasattr(self, '_spin_text2'):
                self._spin_cv.itemconfig(self._spin_text2, text=subtext)

    def _tick_spinner(self):
        if not self._spin_running:
            return
        self._spin_angle = (self._spin_angle + 10) % 360
        if hasattr(self, '_spin_cv') and self._spin_cv.winfo_exists():
            self._spin_cv.itemconfig('sp_arc', start=self._spin_angle)
        self._spin_after_id = self.root.after(28, self._tick_spinner)

    def _destroy_spinner_card(self):
        self._spin_running = False
        if self._spin_after_id:
            self.root.after_cancel(self._spin_after_id)
            self._spin_after_id = None
        if hasattr(self, '_spin_cv') and self._spin_cv.winfo_exists():
            self._spin_cv.destroy()

    # ── Actions ───────────────────────────────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path = path
            self.raw_image = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_image(self.raw_image)
            self._enable_run_btn()
            self._hide_save_btn()
            self._panel_show_empty()
            self._set_status('图片载入成功', TEXT_PRI, ACCENT_PUR)

    def run_analysis(self):
        if self.raw_image is None:
            return
        self._hide_toggle_btn()
        self._panel_show_empty()
        self.result_img = None
        self.import_btn.config(state='disabled')
        self.run_btn.config(state='disabled')
        self._set_status('正在分析中...', ACCENT_YELLOW, ACCENT_YELLOW)
        self._start_scan()
        self._start_pulse()
        self._build_spinner_card(
            text='正在估算细胞直径...',
            subtext='diameter=0 自动估算')
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        t = threading.Thread(target=self._analysis_worker, daemon=True)
        t.start()
        self._check_result()

    def _analysis_worker(self):
        try:
            # Pass 1: estimate diameter
            masks_est, _, _ = self.model.eval(self.raw_image, diameter=0, channels=[0, 0])
            from cellpose.utils import diameters
            diam = float(diameters(masks_est)[0])  # returns (median, per_cell_array)
            if diam == 0 or np.isnan(diam):
                diam = 30.0
            print(f"Cellpose 估算直径: {diam:.1f}px")

            H, W = self.raw_image.shape[:2]
            short_side = min(H, W)
            scale = short_side / 1080.0
            corrected_diam = max(20.0, min(diam * scale, 500.0))
            print(f"校正后直径: {corrected_diam:.1f}px  (短边={short_side}px, scale={scale:.2f})")

            # Pass 2: segmentation with corrected diameter
            self.root.after(0, self._set_spinner_text,
                            '正在分割识别...',
                            f'diameter={corrected_diam:.1f}px')

            masks = self.model.eval(
                self.raw_image,
                diameter=corrected_diam,
                channels=[0, 0],
                flow_threshold=0.95,
                cellprob_threshold=1.0,
                min_size=int(corrected_diam ** 2 * 0.2),
                resample=True,
            )[0]
            self._result_queue.put(('ok', masks))
        except Exception as e:
            self._result_queue.put(('err', e))

    def _check_result(self):
        try:
            status, payload = self._result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_result)
            return
        self._stop_scan()
        self._stop_pulse()
        self._destroy_spinner_card()
        self.import_btn.config(state='normal')
        self._enable_run_btn()
        if status == 'ok':
            self.render_results(payload)
        else:
            messagebox.showerror("识别失败", f"{payload}")

    # ── Render ────────────────────────────────────────────────────────────────
    def render_results(self, masks):
        res_img  = self.raw_image.copy()
        gray     = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        cell_ids = np.unique(masks)[1:]
        total_detected = len(cell_ids)

        # ── Step 1: completeness ──────────────────────────────────────────────
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
            circle_area = math.pi * er * er
            circle_comp = mask_area / circle_area if circle_area > 0 else 0.0

            completeness = circle_comp if touches else hull_comp
            method       = "circle"    if touches else "hull"

            print(f"  cell {cid}: mask={mask_area:.0f}px  hull={hull_comp:.3f}"
                  f"  circle={circle_comp:.3f}  touches={'Y' if touches else 'N'}"
                  f"  -> {method}={completeness:.3f}")

            if completeness < 0.85:
                print(f"  [skip] cell {cid}: {method}={completeness:.3f} < 0.85")
                continue

            perimeter = cv2.arcLength(contours[0], True)
            candidates.append({
                "cid": cid, "mask": mask, "mask_area": mask_area,
                "contours": contours, "perimeter": perimeter,
                "er": er,
            })
        print(f"[Step1] 完整度过滤后: {len(candidates)} / {total_detected}")

        # ── Step 1b: dark region filter ───────────────────────────────────────
        filtered_dark = []
        for c in candidates:
            pixels = gray[c["mask"] > 0]
            mean_brightness = float(pixels.mean()) if len(pixels) > 0 else 0.0
            if mean_brightness < 15:
                print(f"  [skip-dark] cell {c['cid']}: mean_brightness={mean_brightness:.1f} < 15")
                continue
            filtered_dark.append(c)
        candidates = filtered_dark
        print(f"[Step1b] 暗区过滤后: {len(candidates)}")

        # ── Step 2: area ──────────────────────────────────────────────────────
        if candidates:
            median_area = float(np.median([c["mask_area"] for c in candidates]))
            area_thresh = median_area * 0.15
            before2     = len(candidates)
            filtered    = []
            for c in candidates:
                if c["mask_area"] < area_thresh:
                    print(f"  [skip-area] cell {c['cid']}: "
                          f"area={c['mask_area']:.0f} < thresh={area_thresh:.0f}")
                    continue
                filtered.append(c)
            candidates = filtered
            print(f"[Step2] 面积过滤后: {len(candidates)} / {before2}  "
                  f"(中位数={median_area:.0f}, 阈值={area_thresh:.0f})")

        # ── Step 3: circularity ───────────────────────────────────────────────
        filtered = []
        before3  = len(candidates)
        for c in candidates:
            circ = (4 * math.pi * c["mask_area"] / (c["perimeter"] ** 2)
                    if c["perimeter"] > 0 else 0.0)
            if circ < 0.5:
                print(f"  [skip-circ] cell {c['cid']}: circularity={circ:.3f} < 0.50")
                continue
            filtered.append(c)
        candidates = filtered
        print(f"[Step3] 圆形度过滤后: {len(candidates)} / {before3}")

        # ── Brightness ────────────────────────────────────────────────────────
        cell_list = []
        H_img, W_img = gray.shape
        for c in candidates:
            M = cv2.moments(c["mask"])
            if M["m00"] <= 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            ir = max(1, int(c["er"] * 0.8))

            rx1 = max(0, cx - ir);  ry1 = max(0, cy - ir)
            rx2 = min(W_img, cx + ir + 1);  ry2 = min(H_img, cy + ir + 1)

            roi_h, roi_w = ry2 - ry1, rx2 - rx1
            inner_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.circle(inner_mask, (cx - rx1, cy - ry1), ir, 1, -1)

            sample_mask = (inner_mask > 0) & (c["mask"][ry1:ry2, rx1:rx2] > 0)
            cell_pixels = gray[ry1:ry2, rx1:rx2][sample_mask]

            if len(cell_pixels) > 50:
                k = max(1, int(len(cell_pixels) * 0.05))
                peak = float(np.mean(np.partition(cell_pixels, -k)[-k:]))
                cell_list.append({
                    "brightness": peak,
                    "pos": (cx, cy),
                    "contours": c["contours"],
                    "mask": c["mask"],
                    "er": c["er"],
                })

        print(f"过滤前细胞数: {total_detected}  最终有效细胞: {len(cell_list)}")
        cell_list.sort(key=lambda x: x["brightness"], reverse=True)
        self.cell_list = cell_list

        if cell_list:
            top3   = cell_list[:3]
            others = cell_list[3:]

            # Yellow contours + rank number for non-top-3
            for idx, cell in enumerate(others, start=4):
                cx, cy = cell["pos"]
                cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 255), 2)
                cv2.putText(res_img, str(idx), (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            # Ring overlay for ALL valid cells
            for cell in cell_list:
                cx, cy = cell["pos"]
                er_int  = max(1, int(cell["er"]))
                ir_int  = max(1, int(cell["er"] * 0.8))
                outer_m = np.zeros((H_img, W_img), dtype=np.uint8)
                inner_m = np.zeros((H_img, W_img), dtype=np.uint8)
                cv2.circle(outer_m, (cx, cy), er_int, 1, -1)
                cv2.circle(inner_m, (cx, cy), ir_int, 1, -1)
                ring = (outer_m > 0) & (inner_m == 0) & (cell["mask"] > 0)
                res_img[ring] = (res_img[ring] * 0.6).astype(np.uint8)

            # Green contour + center dot + label for top 3
            for rank, cell in enumerate(top3, start=1):
                cx, cy = cell["pos"]
                cv2.drawContours(res_img, cell["contours"], -1, (0, 255, 0), 2)
                cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
                label = f"{rank} | {int(cell['brightness'])}"
                cv2.putText(res_img, label, (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"  #{rank}  坐标=({cx}, {cy})  亮度={int(cell['brightness'])}")

            top1 = top3[0]
            cx1, cy1 = top1["pos"]
            self._set_status(
                f'完成 · 检测: {total_detected} / 有效: {len(cell_list)}'
                f' · #1 ({cx1}, {cy1}) 亮度={int(top1["brightness"])}',
                ACCENT_GREEN, ACCENT_GREEN)
        else:
            self._set_status(
                f'完成 · 检测: {total_detected} / 有效: 0',
                TEXT_SEC, TEXT_DIM)

        self.result_img = res_img
        self.display_image(res_img)
        self._show_toggle_btn()
        self._panel_populate(cell_list)

    # ── Data panel (fixed right side) ─────────────────────────────────────────
    def _build_data_panel(self):
        # Title row
        title_row = tk.Frame(self.right_frame, bg=PANEL_BG, padx=12, pady=9)
        title_row.pack(fill='x', side='top')
        self._panel_title = tk.Label(
            title_row, text='细胞数据',
            bg=PANEL_BG, fg=ACCENT_PUR, font=UI_FONT)
        self._panel_title.pack(side='left')

        tk.Frame(self.right_frame, bg=BORDER, height=1).pack(fill='x', side='top')

        # Text widget + scrollbar
        text_area = tk.Frame(self.right_frame, bg=BG_TITLEBAR)
        text_area.pack(fill='both', expand=True, side='top')

        sb = tk.Scrollbar(text_area, orient='vertical')
        sb.pack(side='right', fill='y')

        self._data_text = tk.Text(
            text_area,
            font=PANEL_FONT,
            bg=BG_TITLEBAR,
            fg=TEXT_SEC,
            state='disabled',
            relief='flat',
            bd=0,
            highlightthickness=0,
            cursor='arrow',
            wrap='none',
            spacing1=3,
            spacing3=3,
            yscrollcommand=sb.set,
        )
        self._data_text.pack(side='left', fill='both', expand=True, padx=6)
        sb.config(command=self._data_text.yview)

        self._data_text.tag_configure(
            'header', foreground=ACCENT_PUR,
            font=(PANEL_FONT[0], PANEL_FONT[1], 'bold'))
        self._data_text.tag_configure('sep',    foreground=TEXT_DIM)
        self._data_text.tag_configure('top3',   foreground=ACCENT_GREEN)
        self._data_text.tag_configure('normal', foreground=TEXT_SEC)

        self._panel_show_empty()

    def _panel_show_empty(self):
        self._panel_title.config(text='细胞数据')
        self._data_text.config(state='normal')
        self._data_text.delete('1.0', 'end')
        self._data_text.insert('end', '\n  等待识别...', 'normal')
        self._data_text.config(state='disabled')

    def _panel_populate(self, cell_list):
        self._panel_title.config(text=f'细胞数据  {len(cell_list)} 个')
        self._data_text.config(state='normal')
        self._data_text.delete('1.0', 'end')

        hdr = (f"{'排名':<4} {'坐标X':<7} {'坐标Y':<7}"
               f" {'直径px':<8} {'亮度':<6}")
        self._data_text.insert('end', hdr + '\n', 'header')
        self._data_text.insert('end', '─' * 38 + '\n', 'sep')

        for i, cell in enumerate(cell_list, start=1):
            cx, cy = cell["pos"]
            bv     = int(cell["brightness"])
            diam   = f'{cell["er"] * 2:.1f}'
            rank   = f'#{i}'
            line   = (f"{rank:<4} {cx:<7} {cy:<7}"
                      f" {diam:<8} {bv:<6}")
            tag    = 'top3' if i <= 3 else 'normal'
            self._data_text.insert('end', line + '\n', tag)

        self._data_text.config(state='disabled')

    # ── Save ──────────────────────────────────────────────────────────────────
    def save_results(self):
        if not self.cell_list or self.image_path is None or self.result_img is None:
            return

        if hasattr(sys, '_MEIPASS'):
            prog_dir = os.path.dirname(sys.executable)
        else:
            prog_dir = os.path.dirname(os.path.abspath(__file__))

        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        save_dir = os.path.join(prog_dir, '细胞数据', stem)
        os.makedirs(save_dir, exist_ok=True)

        cv2.imencode('.png', self.result_img)[1].tofile(
            os.path.join(save_dir, f"{stem}_result.png"))

        with open(os.path.join(save_dir, f"{stem}_top3.csv"),
                  'w', encoding='utf-8') as f:
            f.write("排名,坐标X,坐标Y,亮度值\n")
            for i, cell in enumerate(self.cell_list[:3], start=1):
                cx, cy = cell["pos"]
                f.write(f"{i},{cx},{cy},{int(cell['brightness'])}\n")

        with open(os.path.join(save_dir, f"{stem}_data.csv"),
                  'w', encoding='utf-8') as f:
            f.write("编号,直径(px),坐标X,坐标Y,亮度值\n")
            for i, cell in enumerate(self.cell_list, start=1):
                cx, cy = cell["pos"]
                diameter = round(cell["er"] * 2, 1)
                f.write(f"{i},{diameter},{cx},{cy},{int(cell['brightness'])}\n")

        self._set_status(f'已保存到 细胞数据/{stem}/', ACCENT_GREEN, ACCENT_GREEN)

    # ── Display ───────────────────────────────────────────────────────────────
    def display_image(self, img):
        self.canvas.delete('empty_state')
        self.canvas.delete('cell_image')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        cw = self.canvas.winfo_width()  or 820
        ch = self.canvas.winfo_height() or 714
        ratio = min(cw / w, ch / h)
        nw, nh = int(w * ratio), int(h * ratio)
        img_pil = Image.fromarray(img_rgb).resize(
            (nw, nh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(
            cw // 2, ch // 2, image=self.tk_img,
            anchor='center', tags='cell_image')


if __name__ == "__main__":
    if ttk is not None:
        root = ttk.Window(themename='darkly')
    else:
        root = tk.Tk()
    app = CellAppCP3(root)
    root.mainloop()
