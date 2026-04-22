# 完成报告

---

## 当前版本：v2.1.0（2026-04-22）

### 完成内容

#### 1. 右侧固定数据面板

- 窗口右侧固定 280px 面板，始终可见
- 识别前显示"等待识别..."空状态
- 识别后填充全部有效细胞数据，Top 3 绿色高亮，其余灰白色
- 切换原图/结果图时面板保留不隐藏
- 列结构：排名 / X / Y / 直径px / 亮度，像素级对齐（Frame + pack_propagate=False）
- 数据多时可滚动（Canvas + Scrollbar）

#### 2. 全细胞外圈遮罩

- 环形压暗遮罩从 top3 扩展到所有有效细胞
- 外圈 20% 区域压暗至 60% 亮度

#### 3. CSV 数据导出

| 文件 | 内容 |
|------|------|
| `_result.png` | 含标注的结果图 |
| `_top3.csv` | 排名,坐标X,坐标Y,亮度值 |
| `_data.csv` | 编号,直径(px),坐标X,坐标Y,亮度值（新增亮度列）|

#### 4. Bug 修复

- cv2.putText 不支持中文·字符 → 改为 ASCII `|`
- 面积阈值 10% → 15%，圆形度阈值 0.4 → 0.5，减少小杂质误识别

#### 5. 窗口调整

- 宽度 1100 → 1400px，固定大小
- 移除最大化按钮，标题栏只保留最小化和关闭

---

## 历史记录

### v2.0.0（2026-04-22）

OLED dark cinema 视觉重设计，异步模型加载，原图/结果切换，queue 轮询修复。

### v1.1.0（2026-04-21）

边缘完整度 ≥ 90% 过滤，面积/圆形度过滤，轮廓描边标注。

### v1.0.0（2026-04-20）

基础 Cellpose 细胞识别，边缘截断过滤，GitHub Actions 打包流程。

---

## Release 链接

- [v2.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v2.0.0)
- [v1.2.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.2.0)
- [v1.1.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.0.0)
