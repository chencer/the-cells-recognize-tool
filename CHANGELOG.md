# Changelog

## v2.1.0 — 2026-04-22

- feat: 状态栏新增细胞总数统计，格式 `完成 · 检测: XX / 有效: XX · #1 (x, y) 亮度=XXX`
- feat: 保存按钮，分析完成后出现在工具栏，导入新图时自动隐藏
- feat: 保存输出 `细胞数据/原文件名/` 目录，包含结果图、top3 坐标、所有细胞直径数据
- feat: Top 3 标注格式改为 `序号 · 亮度值`（如 `1 · 247`）
- feat: 其余有效细胞标注白色小字编号（从 4 开始）

## v2.0.0 — 2026-04-22

- redesign: OLED dark cinema + glassmorphism 全局视觉重设计
- feat: 环形采样区域可视化，直观展示亮度计算范围
- feat: 原图 / 结果图切换按钮
- feat: 异步加载 Cellpose 模型，启动时显示进度动画
- feat: Windows 原生风格标题栏 + 浮动工具栏
- fix: queue 轮询机制修复，消除 UI 线程阻塞

## v1.2.0 — 2026-04-21

- feat: 亮度 ROI 优化，改为 80% 内圈前 10% 最亮像素均值
- feat: Top 3 标注，最亮三个细胞分级高亮显示
- feat: Variant B 沉浸式 UI 重构（macOS 标题栏 + 动效）
- feat: 重构 UI 为黑紫沉浸风格

## v1.1.0 — 2026-04-21

- feat: 面积过滤 + 圆形度过滤，去除噪声和异形物体
- fix: 轮廓描边替代圆圈标注，视觉更准确
- fix: 边缘完整度阈值提高到 ≥ 90%
- fix: 修复边界截断细胞过滤逻辑

## v1.0.0 — 2026-04-20

- feat: 基础细胞识别，集成 Cellpose cyto3 模型
- feat: 过滤边缘截断细胞（边界像素接触检测）
- feat: 输出最亮细胞坐标与亮度值
- ci: GitHub Actions 自动打包 Windows exe
- fix: 兼容 PowerShell heredoc、PyInstaller 弯引号、Windows cp1252 乱码
