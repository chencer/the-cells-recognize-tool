# The Cells Recognize Tool

基于 **Cellpose v3.0** 的荧光细胞图像分割与亮度排名桌面工具。

---

## 功能概览

- 导入荧光显微镜图片，一键完成细胞分割
- 自动过滤边缘截断的不完整细胞
- 按**核心亮度**（前 10% 高亮像素均值）对所有细胞排名
- 在图像上高亮标注最亮细胞，并输出其**坐标与亮度值**
- 状态栏实时显示：`最亮细胞: 坐标(x, y)  亮度=xxx`

---

## 文件结构

```
cells_find.py       # 主程序（打包版，内置本地模型加载）
cells_recognize.py  # 开发版（直接调用系统 cellpose）
cyto3               # 本地 Cellpose cyto3 模型权重
requirements.txt    # 依赖列表
.github/workflows/  # GitHub Actions 自动打包为 .exe
```

---

## 环境依赖

```bash
pip install -r requirements.txt
```

核心依赖：

| 包 | 版本 |
|---|---|
| cellpose | 3.0.11 |
| opencv-python | ≥ 4.8 |
| torch | ≥ 2.0 |
| Pillow | ≥ 10.0 |

---

## 使用方法

```bash
python cells_find.py
```

1. 点击 **1. 导入图片** 选择荧光图像
2. 点击 **2. 细胞识别** 开始分割
3. 结果图像中：
   - **红色轮廓 + 标签**：最亮的 2 个细胞
   - **黄色轮廓**：其余识别细胞
   - 状态栏显示 Top 1 细胞坐标与亮度值

---

## 排名算法

每个细胞的亮度得分 = 该细胞区域内**前 10% 最亮像素的均值**。

相比全区域平均值，此方法能准确识别小而亮的细胞，避免大而暗的细胞因面积优势虚排前列。

---

## 边缘过滤

识别完成后自动剔除与图像任意边界相交的细胞——这类细胞因拍摄视野截断而形态不完整，不参与排名。

---

## 打包为 .exe

推送到 `main` 分支后，GitHub Actions 自动触发打包流程，产物见 Releases。

---

## License

MIT
