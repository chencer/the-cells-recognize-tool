# 完成报告

---

## v2.1.0 — 2026-04-22

### 新增功能

#### 1. 细胞总数统计（状态栏）

**修改位置**：`render_results` → `self._set_status`

**新格式**：
```
完成 · 检测: 42 / 有效: 38 · #1 (682, 291) 亮度=247
```
- `检测` = Cellpose 识别的原始细胞总数（过滤前）
- `有效` = 经完整度 + 面积 + 圆形度过滤后保留数量

#### 2. 保存按钮

**修改位置**：`_build_floating_toolbar`、`_show_toggle_btn`、`_hide_toggle_btn`

- 绿色按钮，与「原图」切换按钮并排
- 分析完成后随切换按钮一起出现，导入新图时自动隐藏

#### 3. 数据保存逻辑

**新增方法**：`save_results`

保存路径：程序所在目录 / `细胞数据` / 原文件名（无扩展名）

| 文件 | 内容 |
|------|------|
| `_result.png` | 含标注的结果图 |
| `_top3.txt` | 亮度前三：排名、坐标、亮度值 |
| `_data.txt` | 所有有效细胞：编号、直径、坐标（按亮度排序） |

直径 = `cell["er"] * 2`（最小外接圆直径，来自 Step 1 已计算的 `er`）

#### 4. 结果图标注增强

**修改位置**：`render_results` 标注循环

- Top 3：标注 `rank · brightness`（如 `1 · 247`），绿色，字号 0.7
- 其余细胞：标注白色小字编号（4, 5, 6…），字号 0.5

### 新增实例变量

| 变量 | 类型 | 用途 |
|------|------|------|
| `self.image_path` | `str \| None` | 当前图片路径，供保存时提取文件名 |
| `self.cell_list` | `list \| None` | 分析结果，供保存按钮读取 |

---

## v1.1.0 — 2026-04-21（历史记录）

### Bug 1：边缘截断细胞误判
- **原因**：凸包比方法对圆弧截断无效（弧形本身是凸形，比值 ≈ 1.0）
- **修复**：改用边界像素接触检测

### Bug 2：标注颜色错误
- **修复**：最亮细胞绿色轮廓，其余细胞黄色轮廓

## Release 链接

- [v2.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v2.0.0)
- [v1.2.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.2.0)
- [v1.1.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.0.0)
