# The Cells Recognize Tool

基于 **Cellpose cyto3** 模型的荧光细胞图像分割与亮度排名桌面工具。当前版本：**v2.0.0**

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 细胞分割 | Cellpose cyto3 模型，一键完成荧光图像分割 |
| 边缘完整度过滤 | 剔除与图像边界接触的截断细胞（完整度 ≥ 90%） |
| 面积 + 圆形度过滤 | 去除噪声点和异形非细胞物体 |
| 亮度排名 | 80% 内圈前 10% 最亮像素均值，避免大细胞虚排 |
| Top 3 标注 | 最亮三个细胞分级高亮，环形采样区域可视化 |
| 原图 / 结果图切换 | 一键对比原始图像与分析结果 |
| 异步加载 | 启动时后台加载模型，进度动画实时反馈 |
| UI 风格 | OLED dark cinema + glassmorphism，Windows 原生标题栏 |

---

## 使用方法

**Windows 用户**（推荐）：

1. 从 [Releases](https://github.com/chencer/the-cells-recognize-tool/releases) 下载最新 `.exe`
2. 双击运行，等待模型加载完成（首次约 10–30 秒）
3. 点击 **导入图片** 选择荧光图像（.tif / .png / .jpg）
4. 点击 **细胞识别** 开始分割
5. 查看结果：Top 3 细胞高亮标注，右侧显示亮度排名

**开发运行**：

```bash
pip install -r requirements.txt
python cells_find.py
```

---

## 过滤算法说明

### 边缘完整度过滤

检测细胞 mask 的任意像素是否触碰图像四条边（第 0 行/列、最后行/列）。接触则判定为边缘截断细胞并剔除。

> 注：原凸包比方法（mask 面积 / 凸包面积）无法识别圆弧截断，因为弧形 cap 本身即为凸形，比值恒 ≈ 1.0。

### 面积 + 圆形度过滤

- **面积过滤**：去除面积极小的噪声点
- **圆形度** = 4π × 面积 / 周长²，圆形度过低的异形物体不参与排名

### 亮度得分

取细胞区域内 **前 10% 最亮像素的均值**作为亮度得分。相比全区域平均，此方法能准确识别小而亮的目标，避免大而暗的细胞因面积优势虚排前列。

---

## 文件结构

```
cells_find.py         # 主程序（UI + 分析逻辑）
cells_recognize.py    # 开发版（无 UI，直接调用 cellpose）
cyto3                 # 本地 Cellpose cyto3 模型权重
requirements.txt      # 依赖列表
.github/workflows/    # GitHub Actions 自动打包 .exe
```

---

## 环境依赖

```bash
pip install -r requirements.txt
```

| 包 | 版本 |
|----|------|
| cellpose | ≥ 3.0 |
| opencv-python | ≥ 4.8 |
| torch | ≥ 2.0 |
| ttkbootstrap | ≥ 1.10 |
| Pillow | ≥ 10.0 |

---

## License

MIT
