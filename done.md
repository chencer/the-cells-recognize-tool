# 完成报告

---

## 当前版本：v2.1.1（2026-04-23）✅

### 所有已完成功能

| 功能 | 状态 |
|------|------|
| Cellpose cyto3 细胞分割 | ✅ |
| 边缘完整度过滤（≥ 90%） | ✅ |
| 面积过滤（中位数 × 15%） | ✅ |
| 圆形度过滤（≥ 0.5） | ✅ |
| 亮度排名（80% 内圈前 5% 最亮像素） | ✅ |
| 全细胞外圈 20% 压暗遮罩 | ✅ |
| Top 3 绿色标注 + `序号 \| 亮度` | ✅ |
| 其余细胞黄色轮廓 + 白色编号 | ✅ |
| 细胞总数统计（检测 / 有效） | ✅ |
| 右侧固定数据面板（Courier New 10，tk.Text） | ✅ |
| 原图 / 结果图切换 | ✅ |
| CSV 数据导出（top3.csv + data.csv） | ✅ |
| 异步模型加载 + 进度动画 | ✅ |
| Windows 原生标题栏 + 浮动工具栏 | ✅ |
| OLED dark cinema + glassmorphism UI | ✅ |
| 固定窗口大小（1400×750） | ✅ |
| GitHub Actions 自动打包 .exe | ✅ |

---

## v2.1.0 本次新增/修复

### 新增

1. **右侧数据面板**：固定 280px，tk.Text + Courier New 10，f-string 固定宽度对齐，Top 3 绿色，其余灰白，支持滚动
2. **全细胞外圈遮罩**：从 top3 扩展到所有有效细胞
3. **CSV 导出**：`_top3.csv`（排名/X/Y/亮度）、`_data.csv`（编号/直径/X/Y/亮度）
4. **细胞总数统计**：状态栏格式 `完成 · 检测: XX / 有效: XX · #1 (x, y) 亮度=XXX`
5. **保存按钮**：分析后出现在工具栏，导出结果图 + 两个 CSV 文件

### 修复

- cv2.putText 中文间隔号显示问号 → 改为 ASCII `|`
- 小杂质误识别 → 面积阈值 10% → 15%，圆形度阈值 0.4 → 0.5
- 数据面板文字挤压 → 改用 tk.Text 等宽字体渲染

---

---

## v2.1.1 本次修复（2026-04-23）

### 修复内容

- 超宽拼接图识别失败（拼接黑边导致自动直径估算偏小）→ 改为 diameter=120 固定
- 移除不稳定的自动估算和 tiling 分块逻辑
- 圆形度过滤阈值 0.5 → 0.3，减少正常细胞误过滤
- 添加超宽测试图 test_images/wide_test.jpg

---

## Release 链接

- [v2.1.1](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v2.1.1)（当前最新）
- [v2.1.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v2.1.0)
- [v2.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v2.0.0)
- [v1.2.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.2.0)
- [v1.1.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/chencer/the-cells-recognize-tool/releases/tag/v1.0.0)
